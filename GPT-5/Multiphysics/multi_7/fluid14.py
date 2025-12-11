# file: ns_brinkman_channel.py
from __future__ import print_function
from dolfin import *
import numpy as np

# -----------------------
# Geometry and mesh
# -----------------------
L, H = 1.0, 0.20
nx, ny = 200, 40  # structured triangulation (robust for Newton)
mesh = RectangleMesh(Point(0.0, 0.0), Point(L, H), nx, ny, "crossed")

# Porous block coordinates
xa, xb = 0.4, 0.6

# -----------------------
# Parameters (SI)
# -----------------------
rho = 1.0         # kg m^-3
mu  = 0.01        # Pa s
K   = 1.0e-6      # m^2
Ubar = 1.0        # m s^-1

# -----------------------
# Subdomain/cell markers: 0 = fluid (Omega_f), 1 = porous (Pi)
# -----------------------
cells = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)

class PorousCells(SubDomain):
    def inside(self, x, on_boundary):
        return (x[0] >= xa - DOLFIN_EPS) and (x[0] <= xb + DOLFIN_EPS)

PorousCells().mark(cells, 1)
dxm = Measure("dx", domain=mesh, subdomain_data=cells)

# -----------------------
# Boundary markers
# -----------------------
left_id, right_id, bot_id, top_id = 1, 2, 3, 4
facets = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
tol = 1e-12

class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, tol)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], L, tol)

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0, tol)

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], H, tol)

Left().mark(facets, left_id)
Right().mark(facets, right_id)
Bottom().mark(facets, bot_id)
Top().mark(facets, top_id)
ds_ = Measure("ds", domain=mesh, subdomain_data=facets)

# -----------------------
# Function spaces (P2-P1)
# -----------------------
V = VectorFunctionSpace(mesh, "Lagrange", 2)
Q = FunctionSpace(mesh, "Lagrange", 1)
mixed_el = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, mixed_el)

# Unknowns/tests
w = Function(W)
(u, p) = split(w)
(v, q) = TestFunctions(W)

# -----------------------
# Inlet profile (Poiseuille)
# -----------------------
uin = Expression(("6.0*Ubar*x[1]*(H - x[1])/(H*H)", "0.0"),
                 degree=2, Ubar=Ubar, H=H)

# -----------------------
# Boundary conditions
# -----------------------
bc_inlet_u = DirichletBC(W.sub(0), uin, facets, left_id)
bc_wall_b  = DirichletBC(W.sub(0), Constant((0.0, 0.0)), facets, bot_id)
bc_wall_t  = DirichletBC(W.sub(0), Constant((0.0, 0.0)), facets, top_id)

# Pressure gauge (fix nullspace) at outlet centre
class GaugePoint(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], L, 1e-10) and near(x[1], H/2.0, 1e-3)
gauge = GaugePoint()
bc_p0 = DirichletBC(W.sub(1), Constant(0.0), gauge, method="pointwise")

bcs = [bc_inlet_u, bc_wall_b, bc_wall_t, bc_p0]

# -----------------------
# Operators
# -----------------------
def eps(u):
    return sym(grad(u))

# Brinkman drag only in porous subdomain: (mu/K) * u
drag_coeff = mu/Constant(K)

# -----------------------
# Variational form (steady NS + Brinkman in Π)
# Momentum: rho*(grad(u)*u):v + 2*mu*eps(u):eps(v) - p*div(v) + (mu/K) u·v (in Π) = 0
# Continuity: q*div(u) = 0
# Outlet traction-free is the natural boundary condition (no extra term).
# -----------------------
F = (
    rho*inner(grad(u)*u, v)*dx
  + 2.0*mu*inner(eps(u), eps(v))*dx
  - p*div(v)*dx
  + q*div(u)*dx
  + drag_coeff*inner(u, v)*dxm(1)     # only in porous block
)

# Jacobian for Newton
J = derivative(F, w, TrialFunction(W))

# -----------------------
# Solve (Newton)
# -----------------------
problem = NonlinearVariationalProblem(F, w, bcs, J)
solver = NonlinearVariationalSolver(problem)
prm = solver.parameters
prm["nonlinear_solver"] = "newton"
prm["newton_solver"]["relative_tolerance"] = 1e-8
prm["newton_solver"]["absolute_tolerance"] = 1e-10
prm["newton_solver"]["maximum_iterations"] = 50
prm["newton_solver"]["linear_solver"] = "mumps"
solver.solve()

u_sol, p_sol = w.split(deepcopy=True)

# -----------------------
# Outputs
# -----------------------

# 1) Save (u,p) to XDMF
xdmf = XDMFFile(mesh.mpi_comm(), "q14_solution.xdmf")
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True
u_sol.rename("u", "velocity")
p_sol.rename("p", "pressure")
xdmf.write(u_sol, 0.0)
xdmf.write(p_sol, 0.0)
xdmf.close()
print("Saved (u,p) to q14_solution.xdmf")

# 2) Pressure drop across porous block at centreline (y = H/2)
#    Sample just before/after Π to avoid evaluating exactly on the interface.
y_c = H/2.0
xL = max(xa - 0.01, 1e-9)     # 0.39 if xa=0.4 -> use 0.39
xR = min(xb + 0.01, L-1e-9)   # 0.61 if xb=0.6 -> use 0.61
p_left  = float(p_sol(Point(xL, y_c)))
p_right = float(p_sol(Point(xR, y_c)))
dp = p_left - p_right

with open("q14_dp.txt", "w") as f:
    f.write("p_before = %.9e Pa @ x=%.5f, y=%.5f\n" % (p_left,  xL, y_c))
    f.write("p_after  = %.9e Pa @ x=%.5f, y=%.5f\n" % (p_right, xR, y_c))
    f.write("Delta_p  = %.9e Pa (before - after)\n" % dp)

print("Pressure drop across porous block (centreline): Δp = %.6e Pa" % dp)

# 3) Colour map of speed |u|
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    V1 = FunctionSpace(mesh, "CG", 1)
    speed = project(sqrt(dot(u_sol, u_sol)), V1)

    coords = mesh.coordinates()
    cells_arr = mesh.cells()
    if cells_arr.shape[1] == 3:
        triang = Triangulation(coords[:,0], coords[:,1], cells_arr)
        z = speed.compute_vertex_values(mesh)

        plt.figure(figsize=(10, 10*H/L))
        tpc = plt.tripcolor(triang, z, shading="gouraud")
        # draw porous block overlay
        plt.gca().add_patch(plt.Rectangle((xa, 0.0), xb-xa, H, fill=False, linewidth=1.2))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim(0.0, L); plt.ylim(0.0, H)
        plt.xlabel("x (m)"); plt.ylabel("y (m)")
        cbar = plt.colorbar(tpc); cbar.set_label("|u| (m/s)")
        plt.title("Velocity magnitude |u| with porous block Π outlined")
    else:
        pplt = plot(speed); plt.colorbar(pplt)

    plt.tight_layout()
    plt.savefig("q14_speed.png", dpi=220)
    plt.close()
    print("Saved q14_speed.png")
except Exception as e:
    print("Plotting failed:", e)

# [[END_OF_CODE]]