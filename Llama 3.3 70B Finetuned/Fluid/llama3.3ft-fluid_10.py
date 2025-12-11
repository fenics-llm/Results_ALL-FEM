from dolfin import *
import matplotlib.pyplot as plt

# Geometry
mesh = UnitSquareMesh(32, 32)

# Function spaces
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
ME = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, ME)

# Trial and test functions
w = Function(W)
(u, p) = split(w)
(v, q) = TestFunctions(W)

# Temperature space
T_space = FunctionSpace(mesh, "CG", 1)
T_trial = TrialFunction(T_space)
phi = TestFunction(T_space)

# Parameters
rho = Constant(1.0)
mu = Constant(1.5e-5)
alpha = Constant(2.1e-5)
g_beta = Constant(3.15e-5)
T_ref = Constant(0.5)

# Boundary conditions

class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0) and on_boundary
class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 1) and on_boundary
class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 1) and on_boundary
class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0) and on_boundary

left = Left()
right = Right()
top = Top()
bottom = Bottom()

# Temperature BCs
T_left_bc = DirichletBC(T_space, Constant(1.0), left)
T_right_bc = DirichletBC(T_space, Constant(0.0), right)

# Velocity BCs (no-slip everywhere)
noslip = Constant((0.0, 0.0))
u_bcs = [DirichletBC(W.sub(0), noslip, left),
         DirichletBC(W.sub(0), noslip, right),
         DirichletBC(W.sub(0).sub(1), Constant(0.0), top),
         DirichletBC(W.sub(0).sub(1), Constant(0.0), bottom)]

# Variational formulation

F = (mu * inner(grad(u), grad(v)) - div(v) * p + q * div(u)
     + rho * g_beta * T_trial * v[1]) * dx
a = lhs(F)
L = rhs(F)

# Temperature equation
T_eq = alpha * dot(grad(T_trial), grad(phi)) * dx

A_T = assemble(lhs(T_eq))
b_T = assemble(rhs(T_eq))

for bc in [T_left_bc, T_right_bc]:
    bc.apply(A_T, b_T)

T_sol = Function(T_space)
solve(A_T, T_sol.vector(), b_T)

# Solve the system
A = assemble(a)
b = assemble(L)

for bc in u_bcs:
    bc.apply(A, b)

solve(A, w.vector(), b)
u_h, p_h = w.split()

# Compute Nusselt number
h_avg = assemble(-alpha * dot(grad(T_sol), grad(phi)) * ds(3))
print("Average Nusselt number:", h_avg / 2.1e-5)

# Save plots and solution
plt.figure()
plot(T_sol, title="Temperature")
plt.savefig("q11_T.png")

file = XDMFFile("q11_solution.xdmf")
file.parameters["flush_output"] = True
file.parameters["functions_share_mesh"] = True

file.write(u_h, 0)
file.write(p_h, 0)
file.write(T_sol, 0)