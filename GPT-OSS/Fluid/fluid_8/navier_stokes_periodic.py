# filename: navier_stokes_periodic.py
from fenics import *

# -------------------------------------------------
# Parameters
# -------------------------------------------------
rho = Constant(1.0)          # density [kg/m³]
mu  = Constant(0.01)         # dynamic viscosity [Pa·s]
G   = Constant(1.0)          # body force magnitude [N/m³]
f   = as_vector([G, Constant(0.0)])   # body force vector (G,0)

# -------------------------------------------------
# Geometry and mesh
# -------------------------------------------------
Lx, Ly = 1.0, 0.20            # domain size
nx, ny = 128, 32              # number of elements
mesh = RectangleMesh(Point(0.0, 0.0), Point(Lx, Ly), nx, ny, "crossed")

# -------------------------------------------------
# Periodic boundary condition in x‑direction
# -------------------------------------------------
class PeriodicBoundary(SubDomain):
    """Map the right edge (x = Lx) onto the left edge (x = 0)."""
    def inside(self, x, on_boundary):
        return bool(near(x[0], 0.0) and on_boundary)

    def map(self, x, y):
        y[0] = x[0] - Lx
        y[1] = x[1]

pbc = PeriodicBoundary()

# -------------------------------------------------
# Function spaces (Taylor–Hood)
# -------------------------------------------------
V = VectorFunctionSpace(mesh, "CG", 2, constrained_domain=pbc)   # velocity
Q = FunctionSpace(mesh, "CG", 1, constrained_domain=pbc)       # pressure

# Build the mixed space manually (MixedFunctionSpace is not a class)
W_elem = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, W_elem, constrained_domain=pbc)

# -------------------------------------------------
# Boundary definitions
# -------------------------------------------------
class Walls(SubDomain):
    """No‑slip walls at y = 0 and y = Ly."""
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], 0.0) or near(x[1], Ly))

walls = Walls()

class PressurePoint(SubDomain):
    """Single point where pressure is fixed to zero (at the origin)."""
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and near(x[1], 0.0)

p_point = PressurePoint()

# -------------------------------------------------
# Dirichlet boundary conditions
# -------------------------------------------------
zero_velocity = Constant((0.0, 0.0))
bc_u = DirichletBC(W.sub(0), zero_velocity, walls)          # no‑slip on walls

# Pointwise pressure gauge (method='pointwise' works for scalar spaces)
bc_p = DirichletBC(W.sub(1), Constant(0.0), p_point, method='pointwise')

bcs = [bc_u, bc_p]

# -------------------------------------------------
# Variational formulation (steady Navier–Stokes)
# -------------------------------------------------
w = Function(W)                     # unknown (u,p)
(u, p) = split(w)                   # split for readability
(v, q) = TestFunctions(W)           # test functions

# Weak form
F = ( mu*inner(grad(u), grad(v))*dx
      + rho*dot(dot(u, nabla_grad(u)), v)*dx
      - div(v)*p*dx
      + q*div(u)*dx
      - dot(f, v)*dx )

# -------------------------------------------------
# Solve the nonlinear problem
# -------------------------------------------------
solve(F == 0, w, bcs,
      solver_parameters={"newton_solver":
                         {"relative_tolerance": 1e-6,
                          "absolute_tolerance": 1e-8,
                          "maximum_iterations": 30,
                          "linear_solver": "mumps"}})

# -------------------------------------------------
# Extract solutions and save
# -------------------------------------------------
(u_sol, p_sol) = w.split()

xdmf = XDMFFile(mesh.mpi_comm(), "q9_soln.xdmf")
xdmf.write(u_sol, 0.0)
xdmf.write(p_sol, 0.0)
xdmf.close()