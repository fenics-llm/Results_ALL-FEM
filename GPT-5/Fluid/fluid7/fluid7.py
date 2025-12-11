# file: ns_cylinder_legacy.py
from __future__ import print_function
from dolfin import *
from mshr import Rectangle, Circle, generate_mesh
import numpy as np

# -----------------------
# Problem data (SI units)
# -----------------------
L = 2.2
H = 0.41
R = 0.05
xc, yc = 0.20, 0.20
rho = 1.0          # kg/m^3
mu  = 0.001        # Pa s
Ubar = 0.2         # m/s
D = 2.0*R          # cylinder diameter

# -----------------------
# Mesh (mshr CSG geometry)
# -----------------------
channel = Rectangle(Point(0.0, 0.0), Point(L, H))
cyl     = Circle(Point(xc, yc), R)
domain  = channel - cyl

# Resolution ~ controls element size; increase if needed
mesh = generate_mesh(domain, 96)

# -----------------------
# Boundary markers
# -----------------------
inlet_id, walls_id, cyl_id, outlet_id = 1, 2, 3, 4
facets = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)

tol = 1e-8

class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, tol)

class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], L, tol)

class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], 0.0, tol) or near(x[1], H, tol))

class Cylinder(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near((x[0]-xc)**2 + (x[1]-yc)**2, R**2, 5e-6)

Inlet().mark(facets, inlet_id)
Outlet().mark(facets, outlet_id)
Walls().mark(facets, walls_id)
Cylinder().mark(facets, cyl_id)

ds_ = Measure("ds", domain=mesh, subdomain_data=facets)

# -----------------------
# Function spaces (P2-P1)
# -----------------------
V = VectorFunctionSpace(mesh, "Lagrange", 2)
Q = FunctionSpace(mesh, "Lagrange", 1)
mixed_el = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, mixed_el)


# -----------------------
# Inlet profile (parabolic with mean Ubar)
# u_x(y) = 6*Ubar*y*(H-y)/H^2, u_y = 0
# -----------------------
inlet_profile = Expression(
    ("6.0*Ubar*x[1]*(H - x[1])/(H*H)", "0.0"),
    degree=2, Ubar=Ubar, H=H
)

# -----------------------
# Dirichlet BCs
# -----------------------
# Velocity: inlet profile at inlet; no-slip on walls and cylinder
bc_inlet_u  = DirichletBC(W.sub(0), inlet_profile, facets, inlet_id)
bc_walls_u  = DirichletBC(W.sub(0), Constant((0.0, 0.0)), facets, walls_id)
bc_cyl_u    = DirichletBC(W.sub(0), Constant((0.0, 0.0)), facets, cyl_id)

# Pressure reference: p = 0 at one point on outlet to fix nullspace
# Pick outlet centre
class OutletPoint(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], L, tol) and near(x[1], H/2.0, 1e-3)
outlet_point = OutletPoint()
bc_p_point = DirichletBC(W.sub(1), Constant(0.0), outlet_point, method="pointwise")

bcs = [bc_inlet_u, bc_walls_u, bc_cyl_u, bc_p_point]

# -----------------------
# Variational formulation (steady Navier–Stokes)
# -----------------------
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

# For Newton iteration, we will form residual with Functions and linearise
w = Function(W)
u_, p_ = split(w)

# Kinematics and constitutive parts
def epsilon(u):
    return sym(grad(u))

I = Identity(mesh.geometry().dim())
n = FacetNormal(mesh)

# Nonlinear residual F(w; v, q)
F = (
    rho*inner(grad(u_)*u_, v)*dx            # convection
  + 2.0*mu*inner(epsilon(u_), epsilon(v))*dx
  - p_*div(v)*dx
  + q*div(u_)*dx
)

# Natural traction-free at outlet is the default (no extra Neumann term).

# Jacobian (Gateaux derivative) for Newton
J = derivative(F, w, TrialFunction(W))

# -----------------------
# Solve with Newton
# -----------------------
problem = NonlinearVariationalProblem(F, w, bcs, J)
solver = NonlinearVariationalSolver(problem)
prm = solver.parameters

# Reasonable Newton settings
prm["nonlinear_solver"] = "newton"
prm["newton_solver"]["relative_tolerance"] = 1e-8
prm["newton_solver"]["absolute_tolerance"] = 1e-10
prm["newton_solver"]["maximum_iterations"] = 50
prm["newton_solver"]["linear_solver"] = "mumps"

solver.solve()

(u_sol, p_sol) = w.split(True)

# -----------------------
# Drag force on cylinder and drag coefficient
# F_D = ∫_{Γ_c} (sigma · n)_x ds
# sigma = -p I + mu (grad(u)+grad(u)^T)
# C_D = 2*F_D / (rho * Ubar^2 * D)
# -----------------------
sigma = -p_sol*I + mu*(grad(u_sol) + grad(u_sol).T)
traction = dot(sigma, n)
ex = Constant((1.0, 0.0))
F_D = assemble(dot(traction, ex)*ds_(cyl_id))
C_D = 2.0*F_D/(rho*Ubar*Ubar*D)

print("\n=== Results ===")
print("Drag force F_D  = %.6e N per unit thickness" % F_D)
print("Drag coefficient C_D = %.6f" % C_D)

# -----------------------
# Save solution to XDMF
# -----------------------
xdmf = XDMFFile(mesh.mpi_comm(), "q7_soln.xdmf")
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True
xdmf.write(u_sol, 0.0)
xdmf.write(p_sol, 0.0)
xdmf.close()

# -----------------------
# Make speed colormap and save as q7_speed.png
# -----------------------
try:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    V0 = FunctionSpace(mesh, "CG", 1)
    speed = project(sqrt(dot(u_sol, u_sol)), V0)

    # Plot with tripcolor for better quality
    coords = mesh.coordinates()
    cells = mesh.cells()
    tri = cells if cells.shape[1] == 3 else None  # just in case

    # Build a triangulation plot
    from matplotlib.tri import Triangulation
    if tri is None:
        # If the mesh is not triangles (rare here), fallback to plot()
        p = plot(speed)
        plt.colorbar(p)
    else:
        triang = Triangulation(coords[:,0], coords[:,1], tri)
        z = speed.compute_vertex_values(mesh)
        plt.figure(figsize=(8, 1.6*8/L*H))
        tpc = plt.tripcolor(triang, z, shading="gouraud")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim(0.0, L)
        plt.ylim(0.0, H)
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        cbar = plt.colorbar(tpc)
        cbar.set_label("|u| (m/s)")
        plt.title("Speed magnitude |u|")
    plt.tight_layout()
    plt.savefig("q7_speed.png", dpi=200)
    plt.close()
    print("Saved speed plot to q7_speed.png")
except Exception as e:
    print("Plotting failed:", e)

# -----------------------
# Also save drag/inputs summary
# -----------------------
with open("ns_summary.txt", "w") as f:
    f.write("rho = %.6g kg/m^3\n" % rho)
    f.write("mu  = %.6g Pa s\n" % mu)
    f.write("Ubar= %.6g m/s\n" % Ubar)
    f.write("D   = %.6g m\n" % D)
    f.write("F_D = %.9e N per unit thickness\n" % F_D)
    f.write("C_D = %.9f\n" % C_D)

print("Saved solution to q7_soln.xdmf and summary to ns_summary.txt")
# [[END_OF_CODE]]