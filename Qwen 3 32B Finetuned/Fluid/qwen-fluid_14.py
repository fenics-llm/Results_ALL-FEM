from dolfin import *
from mshr import Rectangle, Circle, generate_mesh
import numpy as np
import matplotlib.pyplot as plt

# Geometry
D = 1.0
mesh = generate_mesh(Rectangle(Point(-30*D, -30*D), Point(30*D, 30*D)) - Circle(Point(0, 0), 0.5*D), 64)

# Function spaces
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
ME = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, ME)
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

# Parameters
U = 1.0
nu = 2.56e-5
rho = 1.0
dt = 0.01
T = 10.0
t = 0.0

# Boundary conditions
inflow = Expression(("U", "0.0"), U=U, degree=2)
outflow = Constant(0.0)
noslip = Constant((0.0, 0.0))
bcs = [
    DirichletBC(W.sub(0), inflow, "near(x[0], -30)"),
    DirichletBC(W.sub(1), outflow, "near(x[0], 30)"),
    DirichletBC(W.sub(0), noslip, "near(x[1], -30) || near(x[1], 30)"),
    DirichletBC(W.sub(0), noslip, "on_boundary && x[0]^2 + x[1]^2 <= 0.25")
]

# Initial conditions
w0 = Function(W)
u0, p0 = w0.split()
u0.vector()[:] = np.random.rand(u0.vector().size()) * 1e-3

# Variational forms
h = CellDiameter(mesh)
tau = h / (2 * nu)
delta = h / (2 * nu)
gamma = 1.0

a = (rho * dot((u - u0)/dt, v) * dx
     + rho * dot(dot(u0, nabla_grad(u)), v) * dx
     + nu * inner(grad(u), grad(v)) * dx
     - div(v) * p * dx
     - div(u) * q * dx
     + tau * rho * dot(dot(u0, nabla_grad(u)), dot(u0, nabla_grad(v))) * dx
     + delta * rho * dot(dot(u0, nabla_grad(u)), div(v)) * dx
     + delta * rho * dot(dot(u0, nabla_grad(v)), div(u)) * dx
     + gamma * div(u) * div(v) * dx)

L = (rho * dot(u0/dt, v) * dx
     + tau * rho * dot(dot(u0, nabla_grad(u0)), dot(u0, nabla_grad(v))) * dx
     + delta * rho * dot(dot(u0, nabla_grad(u0)), div(v)) * dx)

# Time-stepping
w = Function(W)
drag = 0.0
while t < T:
    t += dt
    solve(a == L, w, bcs, solver_parameters={"newton_solver": {"maximum_iterations": 50}})
    u_sol, p_sol = w.split()
    if t >= 8.0:
        drag += assemble(rho * dot(u_sol, Constant((1.0, 0.0))) * ds(0))

# Postprocessing
mean_drag = drag / (T - 8.0)
print("Mean drag coefficient:", mean_drag)

# Save solution
u_sol.rename("velocity", "u")
p_sol.rename("pressure", "p")
with XDMFFile("vms_solution.xdmf") as file:
    file.write(u_sol, T)
    file.write(p_sol, T)