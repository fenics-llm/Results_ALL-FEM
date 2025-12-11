# filename: q2_stokes.py
from dolfin import *
import matplotlib.pyplot as plt

# -------------------------------------------------
# Parameters
# -------------------------------------------------
L = 2.0          # length (m)
H = 0.20         # height (m)
mu = 1.0         # dynamic viscosity (Pa·s)
f = Constant((1.0, 0.0))   # body force

# -------------------------------------------------
# Mesh (120 × 12 uniform rectangles)
# -------------------------------------------------
mesh = RectangleMesh(Point(0.0, 0.0), Point(L, H), 120, 12)

# -------------------------------------------------
# Mixed finite element (Taylor–Hood P2/P1)
# -------------------------------------------------
V_el = VectorElement("Lagrange", mesh.ufl_cell(), 2)   # velocity (P2)
Q_el = FiniteElement("Lagrange", mesh.ufl_cell(), 1)   # pressure (P1)
mixed_el = MixedElement([V_el, Q_el])
W = FunctionSpace(mesh, mixed_el)

# -------------------------------------------------
# Boundary condition: no‑slip on the walls (y = 0 and y = H)
# -------------------------------------------------
class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], 0.0) or near(x[1], H))

walls = Walls()
bc = DirichletBC(W.sub(0), Constant((0.0, 0.0)), walls)   # u = 0 on walls

# -------------------------------------------------
# Variational formulation (steady Stokes)
# -------------------------------------------------
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

a = mu*inner(grad(u), grad(v))*dx - div(v)*p*dx - q*div(u)*dx
L_form = inner(f, v)*dx

# -------------------------------------------------
# Solve the linear system
# -------------------------------------------------
w = Function(W)
solve(a == L_form, w, bcs=[bc], solver_parameters={"linear_solver": "mumps"})

# Split the mixed solution
(u_sol, p_sol) = w.split()

# -------------------------------------------------
# Save solution to XDMF
# -------------------------------------------------
xdmf = XDMFFile(mesh.mpi_comm(), "q2_solution.xdmf")
xdmf.write(u_sol, 0.0)
xdmf.write(p_sol, 0.0)
xdmf.close()

# -------------------------------------------------
# Compute speed magnitude |u|
# -------------------------------------------------
V0 = FunctionSpace(mesh, "Lagrange", 2)
speed = project(sqrt(dot(u_sol, u_sol)), V0)

# -------------------------------------------------
# Plot speed and save as PNG
# -------------------------------------------------
plt.figure(figsize=(8, 2))
c = plot(speed, cmap="viridis")
plt.colorbar(c, label=r"$|\mathbf{u}|$")
plt.title(r"Speed magnitude $|\mathbf{u}|$")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.savefig("q2_speed.png", dpi=300)
plt.close()