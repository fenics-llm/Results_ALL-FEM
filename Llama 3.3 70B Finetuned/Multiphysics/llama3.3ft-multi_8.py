rom dolfin import *

# Geometry: unit square [0, 1] × [0, 1]
mesh = UnitSquareMesh(32, 32)

# Function spaces
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
ME = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, ME)

# Trial and test functions
(U, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

# Parameters
rho = Constant(1.0)
nu = Constant(1e-3)
dt = Constant(5e-4)
t_end = 1.0

# Initial condition for velocity
class U0(UserExpression):
    def eval(self, values, x):
        values[0] = sin(2*pi*x[0])*cos(2*pi*x[1])
        values[1] = -cos(2*pi*x[0])*sin(2*pi*x[1])

# Periodic boundary conditions
class PeriodicBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0) and on_boundary

    def map(self, x, y):
        y[0] = x[0]
        y[1] = 1 - x[1]

# Boundary conditions
u0 = U0(degree=2)
bc = DirichletBC(W.sub(0), u0, PeriodicBoundary())

# Variational formulation
U_n = Function(W)  # previous solution
U_  = Function(W)  # new solution

F = (rho*inner(U - U_n, v)*dx +
     dt*nu*inner(grad(U), grad(v))*dx -
     p*div(v)*dx + div(U)*q*dx)

a = lhs(F)
L = rhs(F)

# Time-stepping loop
t = 0.0
while t < t_end:
    solve(a == L, U_)
    U_n.assign(U_)  # update for next step

    if near(t, [0, 0.25, 0.5, 1.0], 1e-3):
        u, p_ = U_.split(deepcopy=True)
        print(f"Saving at t = {t:.2f}")
        File("velocity_t%.2f.xml.gz" % t) << u

    t += dt(0)
