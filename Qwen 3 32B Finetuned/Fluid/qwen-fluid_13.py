from fenics import *
import mshr

L, H = 1.0, 0.1
mesh = mshr.Rectangle(Point(0, 0), Point(L, H))
mesh = mshr.generate_mesh(mesh, 100*10)

U_max = 0.75
u_expr = Expression(("U_max * 4 * x[1] * (H - x[1]) / pow(H, 2)", "0.0"),
                    U_max=U_max, H=H, degree=2)
u = interpolate(u_expr, VectorFunctionSpace(mesh, "CG", 2))

V = FunctionSpace(mesh, "CG", 1)
c, v = TrialFunction(V), TestFunction(V)
D_val = 1e-5

h = Cell(mesh).radius()
u_mag = sqrt(dot(u, u))
delta = h / (2 * u_mag)
residual = dot(u, grad(c)) - D_val * div(grad(c))

a = (D_val * inner(grad(c), grad(v)) + dot(u, grad(c)) * v
     + delta * inner(residual, dot(u, grad(v)))) * dx

bcs = [
    DirichletBC(V, Constant(0.0), lambda x, on_b: on_b and x[0] < DOLFIN_EPS),
    DirichletBC(V, Constant(1.0), lambda x, on_b: on_b and x[0] > 1.0 - DOLFIN_EPS)
]

c_sol = Function(V)
solve(a == 0, c_sol, bcs)

file = XDMFFile("concentration.xdmf")
file.write(c_sol)
file.close()