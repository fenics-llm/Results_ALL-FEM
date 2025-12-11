# filename: transport_advection_diffusion.py
"""
Steady advection–diffusion of a scalar in a rectangular channel.
Domain: Ω = (0, L) × (0, H),  L = 1.0 m, H = 0.10 m
Mesh: 100 × 10 uniform rectangles
Velocity: u = (U_max·4·y·(H‑y)/H² , 0),  U_max = 0.75 m s⁻¹
Diffusivity: D = 1.0e‑5 m² s⁻¹
BCs:
  inlet  (x = 0) : c = 0
  outlet (x = L) : c = 1
  walls  (y = 0, H) : zero diffusive normal flux (natural Neumann)
Stabilisation: SUPG (Péclet ≈ 750 ≫ 1)
Result saved as XDMF.
"""

from dolfin import *

# ----------------------------------------------------------------------
# Parameters
L = 1.0          # length (m)
H = 0.10         # height (m)
U_max = 0.75     # max velocity (m/s)
D = 1.0e-5       # diffusivity (m^2/s)

# ----------------------------------------------------------------------
# Mesh and function space
mesh = RectangleMesh(Point(0.0, 0.0), Point(L, H), 100, 10, "crossed")
V = FunctionSpace(mesh, "CG", 1)

# ----------------------------------------------------------------------
# Velocity field (Expression uses x[1] for y)
u = Expression(
    ("U_max*4.0*x[1]*(H - x[1]) / (H*H)", "0.0"),
    degree=2,
    U_max=U_max,
    H=H,
)

# ----------------------------------------------------------------------
# Trial / test functions
c = TrialFunction(V)
v = TestFunction(V)

# ----------------------------------------------------------------------
# SUPG stabilization parameters
h = CellDiameter(mesh)                     # element size
u_vec = as_vector((u[0], u[1]))
u_norm = sqrt(dot(u_vec, u_vec) + DOLFIN_EPS)   # avoid division by zero
tau = h / (2.0 * u_norm)                   # classic SUPG tau

# ----------------------------------------------------------------------
# Variational formulation (steady advection‑diffusion with SUPG)
a = ( D * dot(grad(v), grad(c)) * dx
      + dot(u_vec, grad(c)) * v * dx
      + tau * dot(u_vec, grad(v)) * dot(u_vec, grad(c)) * dx )
L_form = Constant(0.0) * v * dx   # no source term

# ----------------------------------------------------------------------
# Boundary conditions
class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and on_boundary

class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], L) and on_boundary

inlet = Inlet()
outlet = Outlet()

bc_inlet = DirichletBC(V, Constant(0.0), inlet)
bc_outlet = DirichletBC(V, Constant(1.0), outlet)
bcs = [bc_inlet, bc_outlet]

# ----------------------------------------------------------------------
# Solve
c_sol = Function(V)
solve(a == L_form, c_sol, bcs,
      solver_parameters={"linear_solver": "mumps"})

# ----------------------------------------------------------------------
# Save solution
xdmf = XDMFFile(mesh.mpi_comm(), "concentration.xdmf")
xdmf.write(c_sol)
xdmf.close()

# ----------------------------------------------------------------------
# Optional: print values at a few locations for quick sanity check
for pt in [(0.0, H/2), (L/2, H/2), (L, H/2)]:
    print(f"c({pt[0]:.3f}, {pt[1]:.3f}) = {c_sol(pt):.6f}")