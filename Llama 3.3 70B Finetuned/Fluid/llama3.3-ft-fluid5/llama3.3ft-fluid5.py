from dolfin import *
import matplotlib.pyplot as plt

# Geometry and mesh
mesh = UnitSquareMesh(128, 128)

# Function space
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
ME = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, ME)

# Boundary markers
left   = AutoSubDomain(lambda x,on_b: on_b and near(x[0],0))
right  = AutoSubDomain(lambda x,on_b: on_b and near(x[0],1))
bottom = AutoSubDomain(lambda x,on_b: on_b and near(x[1],0))
top    = AutoSubDomain(lambda x,on_b: on_b and near(x[1],1))

# Boundary conditions
lid_velocity   = Constant((1.0, 0.0))
noslip         = Constant((0.0, 0.0))
bcs            = [
    DirichletBC(W.sub(0), lid_velocity, top),
    DirichletBC(W.sub(0), noslip, left),
    DirichletBC(W.sub(0), noslip, right),
    DirichletBC(W.sub(0), noslip, bottom)
]

# Pressure gauge
class Origin(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and near(x[1], 0.0)

origin = Origin()
bcs.append(DirichletBC(W.sub(1), Constant(0.0), origin))

# Parameters
rho = 1.0
mu  = 0.01

# Variational problem
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

f = Constant((0.0, 0.0))

a = ( rho*inner(grad(u), grad(v)) - div(v)*p + q*div(u) )*dx
L = inner(f, v)*dx

# Solve
w = Function(W)
solve(a == L, w, bcs)

u_soln, p_soln = w.split()

# Postprocessing
speed = project(sqrt(dot(u_soln, u_soln)), FunctionSpace(mesh, "CG", 1))
plot(speed, title="|u(x)|")
plt.savefig("q5_speed.png")

File("q5_u.pvd") << u_soln
File("q5_p.pvd") << p_soln