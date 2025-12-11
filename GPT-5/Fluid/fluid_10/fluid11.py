# file: boussinesq_cavity_steady.py
from __future__ import print_function
from dolfin import *

# -----------------------
# Geometry, mesh
# -----------------------
Lx, Ly = 1.0, 1.0
nx, ny = 128, 128                  # structured triangular mesh
mesh = RectangleMesh(Point(0.0, 0.0), Point(Lx, Ly), nx, ny, "crossed")

# -----------------------
# Parameters (SI)
# -----------------------
rho   = 1.0                        # kg/m^3
mu    = 1.5e-5                     # Pa s
alpha = 2.1e-5                     # m^2/s  (thermal diffusivity)
gbeta = 3.15e-5                    # m/s^2/K (g * beta)
T_ref = 0.5                        # K

# -----------------------
# Boundary markers
# -----------------------
left_id, right_id, top_id, bot_id = 1, 2, 3, 4
facets = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
tol = 1e-10

class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, tol)
class Right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], Lx, tol)
class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0, tol)
class Top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], Ly, tol)

Left().mark(facets, left_id)
Right().mark(facets, right_id)
Bottom().mark(facets, bot_id)
Top().mark(facets, top_id)

ds_ = Measure("ds", domain=mesh, subdomain_data=facets)

# -----------------------
# Mixed space: (u,p,T) = (P2^2, P1, P1)
# -----------------------
V = VectorFunctionSpace(mesh, "Lagrange", 2)
Q = FunctionSpace(mesh, "Lagrange", 1)
S = FunctionSpace(mesh, "Lagrange", 1)
mixed_el = MixedElement([V.ufl_element(), Q.ufl_element(), S.ufl_element()])
W = FunctionSpace(mesh, mixed_el)

# Unknown and test fields
w  = Function(W)
(u, p, T) = split(w)
(v, q, s) = TestFunctions(W)

# -----------------------
# Boundary conditions
# -----------------------
# Velocity: no-slip on all walls
bc_u_all = DirichletBC(W.sub(0), Constant((0.0, 0.0)), "on_boundary")

# Temperature: Dirichlet on left/right; natural (adiabatic) on top/bottom
bc_T_left  = DirichletBC(W.sub(2), Constant(1.0), facets, left_id)
bc_T_right = DirichletBC(W.sub(2), Constant(0.0), facets, right_id)

# Pressure gauge: p = 0 at a point (0,0)
class GaugePoint(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0, 1e-12) and near(x[1], 0.0, 1e-12)
gauge = GaugePoint()
bc_p0 = DirichletBC(W.sub(1), Constant(0.0), gauge, method="pointwise")

bcs = [bc_u_all, bc_T_left, bc_T_right, bc_p0]

# -----------------------
# Model terms
# -----------------------
def eps(u):
    return sym(grad(u))
I = Identity(2)

# Boussinesq body force: f = (0, rho * g*beta * (T - T_ref))
f = as_vector((Constant(0.0), rho*gbeta*(T - Constant(T_ref))))

# Steady Navier–Stokes + steady advection–diffusion for T
F_mom = rho*inner(grad(u)*u, v)*dx + 2.0*mu*inner(eps(u), eps(v))*dx - p*div(v)*dx - inner(f, v)*dx
F_cont = q*div(u)*dx
F_temp = inner(dot(u, grad(T)), s)*dx + alpha*inner(grad(T), grad(s))*dx

F = F_mom + F_cont + F_temp

# Jacobian for Newton
J = derivative(F, w, TrialFunction(W))

# -----------------------
# Solve nonlinear system (Newton)
# -----------------------
problem = NonlinearVariationalProblem(F, w, bcs, J)
solver = NonlinearVariationalSolver(problem)
prm = solver.parameters
prm["nonlinear_solver"] = "newton"
prm["newton_solver"]["relative_tolerance"] = 1e-9
prm["newton_solver"]["absolute_tolerance"] = 1e-11
prm["newton_solver"]["maximum_iterations"] = 50
prm["newton_solver"]["linear_solver"] = "mumps"
solver.solve()

u_sol, p_sol, T_sol = w.split(deepcopy=True)

# -----------------------
# Average Nusselt number at the left wall
# Here, Nu = - average( dT/dx ) at x=0, with L=1 and ΔT=1.
# Implement as: Nu_avg = ( - ∫_{x=0} ∂T/∂x ds ) / Ly
# -----------------------
dTdx = dot(grad(T_sol), Constant((1.0, 0.0)))  # ∂T/∂x
Nu_int = assemble(-dTdx*ds_(left_id))
Nu_avg = Nu_int / Ly
print("\nAverage Nusselt number at left wall: %.6f" % Nu_avg)

# -----------------------
# Outputs
# -----------------------
# 1) XDMF of (u,p,T)
xdmf = XDMFFile(mesh.mpi_comm(), "q11_solution.xdmf")
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True
u_sol.rename("u", "velocity")
p_sol.rename("p", "pressure")
T_sol.rename("T", "temperature")
xdmf.write(u_sol, 0.0)
xdmf.write(p_sol, 0.0)
xdmf.write(T_sol, 0.0)
xdmf.close()
print("Saved (u,p,T) to q11_solution.xdmf")

# 2) Temperature colour map
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    V0 = FunctionSpace(mesh, "CG", 1)
    T1 = project(T_sol, V0)
    coords = mesh.coordinates()
    cells = mesh.cells()

    if cells.shape[1] == 3:
        triang = Triangulation(coords[:,0], coords[:,1], cells)
        z = T1.compute_vertex_values(mesh)
        plt.figure(figsize=(6, 6))
        tpc = plt.tripcolor(triang, z, shading="gouraud")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim(0.0, Lx); plt.ylim(0.0, Ly)
        plt.xlabel("x (m)"); plt.ylabel("y (m)")
        cbar = plt.colorbar(tpc); cbar.set_label("Temperature T (K)")
        plt.title("Temperature field")
    else:
        p = plot(T1); plt.colorbar(p)

    plt.tight_layout()
    plt.savefig("q11_T.png", dpi=200)
    plt.close()
    print("Saved temperature plot to q11_T.png")
except Exception as e:
    print("Plotting failed:", e)

# Also log summary
with open("q11_summary.txt", "w") as f:
    f.write("rho   = %.6g kg/m^3\n" % rho)
    f.write("mu    = %.6g Pa s\n" % mu)
    f.write("alpha = %.6g m^2/s\n" % alpha)
    f.write("g*beta= %.6g m/s^2/K\n" % gbeta)
    f.write("T_ref = %.6g K\n" % T_ref)
    f.write("Nu_avg(left) = %.9f\n" % Nu_avg)

# [[END_OF_CODE]]