from dolfin import *
import mshr
import matplotlib.pyplot as plt

# Geometry
domain = mshr.Rectangle(Point(0, 0), Point(1, 1))
mesh = mshr.generate_mesh(domain, 64)

# Model parameters
rho, mu, alpha, g, beta = 1.0, 1.5e-5, 2.1e-5, 3.15e-5, 0.5

# Velocity (u), pressure (p) and temperature (T)
class U(PiecewiseConstantElement):
    def value(self, x, ufl_element=None):
        return ufl_expression([0, 0], degree=1)

class P(Expression):
    def eval(self, values, x):
        values[0] = 0.0

class T(Expression):
    def eval(self, values, x):
        values[0] = 1.0 if near(x[0], 0) else 0.0

u = TrialFunction(U)
v = TestFunction(U)
p = TrialFunction(Multifunction(P))
T = TrialFunction(Multifunction(T))

f_body = Expression("rho*g*beta*(T-T_ref)", rho=rho, g=g, beta=beta, T_ref=T_ref)

# Variational forms
a = (mu * dot(grad(u), grad(v)) - div(u) * v) * dx
L_body = f_body * v * dx

a += alpha * inner(grad(T), grad(v)) * dx
L_body += g * beta * rho * T * v * dx

b = p * div(v) * dx
L_body += p * div(v) * dx

# Boundary conditions
bc_left = DirichletBC(U, Constant([0.0, 0.0]), "near(x[0], 0.0)")
bc_right = DirichletBC(U, Constant([0.0, 0.0]), "near(x[0], 1.0)")

# Solve
u_sol, p_sol, T_sol = solve_system(a, L_body, [bc_left, bc_right])

# Postprocessing
left_wall = assemble(inner(u_sol, Constant([1.0, 0.0])) * dx)
print("Left wall Nusselt number: {}".format(left_wall))

# Visualization
plt.figure()
p = plot(T_sol)
plt.colorbar(p)
plt.savefig("q11_T.png")
plt.close()

# Save solution
File("q11_solution.xdmf") << u_sol, p_sol, T_sol