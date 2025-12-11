# filename: navier_stokes_demo.py
from dolfin import *

# Physical parameters
rho = 1.0          # density (kg/m^3)
mu  = 0.01         # dynamic viscosity (Pa·s)
G   = 1.0          # body force magnitude (N/m^3)
f   = Constant((G, 0.0))

# Mesh: rectangular channel [0,1] × [0,0.20] with 128 × 32 cells
mesh = RectangleMesh(Point(0.0, 0.0), Point(1.0, 0.20), 128, 32, "crossed")

# Periodic in x
class PeriodicX(SubDomain):
    def inside(self, x, on_boundary):
        return bool(on_boundary and near(x[0], 0.0) and (not near(x[1], 0.0)) and (not near(x[1], 0.20)))
    def map(self, x, y):
        y[0] = x[0] - 1.0
        y[1] = x[1]

pbc = PeriodicX()

# Taylor–Hood element (P2 velocity, P1 pressure)
Ve = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Pe = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = MixedElement([Ve, Pe])
W  = FunctionSpace(mesh, TH, constrained_domain=pbc)

# No-slip walls (top and bottom)
def walls(x, on_boundary):
    return on_boundary and (near(x[1], 0.0) or near(x[1], 0.20))
bc_walls = DirichletBC(W.sub(0), Constant((0.0, 0.0)), walls)

# Pressure gauge at (0,0) to fix the nullspace
class PointGauge(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and near(x[1], 0.0)
p_gauge = PointGauge()
bc_pressure = DirichletBC(W.sub(1), Constant(0.0), p_gauge, method="pointwise")

bcs = [bc_walls, bc_pressure]

# Unknown and test functions
w = Function(W)                 # (u,p)
(u, p) = split(w)
(v, q) = TestFunctions(W)

# Steady Navier–Stokes residual
R = (2*mu*inner(sym(grad(u)), sym(grad(v))) + rho*inner(dot(u, nabla_grad(u)), v)
     - div(v)*p + q*div(u) - dot(f, v))*dx
J = derivative(R, w)

# Solve the nonlinear problem
solve(J == -R, w, bcs)

# Separate velocity and pressure
(u_sol, p_sol) = w.split(deepcopy=True)

# Write results to XDMF
with XDMFFile(mesh.mpi_comm(), "q9_soln.xdmf") as xdmf:
    xdmf.write(u_sol, 0.0)
    xdmf.write(p_sol, 0.0)