from dolfin import *
from mshr import Rectangle, generate_mesh
import matplotlib.pyplot as plt

L, H = 2.0, 0.2
mesh = generate_mesh(Rectangle(Point(0, 0), Point(L, H)), 160 * 16)

V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
ME = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, ME)
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

mu, rho = 0.01, 1.0
U_bar = 2.5

inlet = Expression(("6*U_bar*(x[1]/H)*(1 - x[1]/H)", "0.0"), U_bar=U_bar, H=H, degree=2)
bcs = [
    DirichletBC(W.sub(0), inlet, "near(x[0], 0.0)"),
    DirichletBC(W.sub(0), Constant((0.0, 0.0)), "near(x[1], 0.0) || near(x[1], H)"),
    DirichletBC(W.sub(0), Constant((0.0, 0.0)), "near(x[0], L)")
]

a = (rho * dot(u, grad(u)) * v * dx
     + mu * inner(grad(u), grad(v)) * dx
     - p * div(v) * dx
     - q * div(u) * dx)
L = Constant(0.0) * v * dx

w = Function(W)
solve(lhs(a) == rhs(L), w, bcs, solver_parameters={"newton_solver": {"relative_tolerance": 1e-6}})
u_sol, p_sol = w.split()

with XDMFFile("q4_soln.xdmf") as file:
    file.write(u_sol, 0.0)
    file.write(p_sol, 0.0)

plt.figure()
plot(u_sol.sub(0))
plt.colorbar()
plt.savefig("q4_ux.png")
