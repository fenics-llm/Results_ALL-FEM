# filename: q1_stokes.py
from dolfin import *
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# Parameters
# --------------------------------------------------------------
L = 2.0          # length  (m)
H = 0.20         # height  (m)
mu = Constant(1.0)   # dynamic viscosity (Pa·s)
p_in = Constant(1.0) # inlet pressure (Pa)
p_out = Constant(0.0)# outlet pressure (Pa)

# --------------------------------------------------------------
# Mesh (100 × 10 uniform rectangles)
# --------------------------------------------------------------
nx, ny = 100, 10
mesh = RectangleMesh(Point(0.0, 0.0), Point(L, H), nx, ny)

# --------------------------------------------------------------
# Function spaces (Taylor–Hood: P2 for velocity, P1 for pressure)
# --------------------------------------------------------------
V_el = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Q_el = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, MixedElement([V_el, Q_el]))

# --------------------------------------------------------------
# Boundary markers
# --------------------------------------------------------------
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)

class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and on_boundary

class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], L) and on_boundary

class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[1], 0.0) or near(x[1], H)) and on_boundary

Inlet().mark(boundaries, 1)   # Γ_in  -> marker 1
Outlet().mark(boundaries, 2)  # Γ_out -> marker 2
Walls().mark(boundaries, 3)   # Γ_w   -> marker 3

ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# --------------------------------------------------------------
# Variational formulation (steady Stokes)
# --------------------------------------------------------------
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

n = FacetNormal(mesh)

a = (mu*inner(grad(u), grad(v)) - div(v)*p + q*div(u))*dx
L_form = -p_in*inner(n, v)*ds(1) - p_out*inner(n, v)*ds(2)   # natural traction BCs

# No‑slip on walls (essential BC)
noslip = Constant((0.0, 0.0))
bc_walls = DirichletBC(W.sub(0), noslip, boundaries, 3)

# --------------------------------------------------------------
# Solve the linear system
# --------------------------------------------------------------
w = Function(W)
solve(a == L_form, w, bcs=[bc_walls],
      solver_parameters={'linear_solver': 'mumps'})

# Split the mixed solution
(u_sol, p_sol) = w.split()

# --------------------------------------------------------------
# Save solution to XDMF (velocity and pressure)
# --------------------------------------------------------------
with XDMFFile(mesh.mpi_comm(), "q1_soln.xdmf") as xdmf:
    xdmf.write(u_sol)   # velocity
    xdmf.write(p_sol)   # pressure

# --------------------------------------------------------------
# Compute speed magnitude |u| and plot it
# --------------------------------------------------------------
V0 = FunctionSpace(mesh, "CG", 2)          # scalar space for speed
speed = project(sqrt(dot(u_sol, u_sol)), V0)

plt.figure(figsize=(8, 2))
p = plot(speed, title=r"Speed $|u|$", cmap="viridis")
plt.colorbar(p)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.tight_layout()
plt.savefig("q1_speed.png", dpi=300)
plt.close()