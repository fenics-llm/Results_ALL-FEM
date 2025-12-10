from dolfin import *
import mshr

# Geometry
domain = mshr.Rectangle(Point(0, 0), Point(100, 180))
hole = mshr.Circle(Point(50, 90), 50)
mesh = mshr.generate_mesh(domain - hole, 64)

# Material
lambda_, mu = 19.44e9, 29.17e9
sigma_y = 243e6

def epsilon(u):
    return sym(grad(u))

def sigma(e):
    return (lambda_ + 2*mu/3)*tr(e)*Identity(2) + 2*mu*e

# Variational problem
u = TrialFunction(V)
v = TestFunction(V)

f = Constant((0.0, -9.81))
a = inner(sigma(epsilon(u)), epsilon(v))*dx
L = dot(f, v)*dx

bc_x = DirichletBC(V, Constant((0.0, 1.0)), "near(x[0], 0.0)")
bc_y = DirichletBC(V, Constant((0.0, 0.0)), "near(x[1], 180.0)")

# Solve
u_sol = Function(V)
solve(a == L, u_sol, [bc_x, bc_y])

# Postprocessing
plot(u_sol, mode="displacement", title="Problem 2")
plt.savefig("problem2.png")
