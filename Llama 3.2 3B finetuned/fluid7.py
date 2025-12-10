from mshr   import Circle, Rectangle, generate_mesh
import dolfin as df

# Geometry
Ω = Rectangle(Point(0, 0), Point(2.2, 0.41)) - Circle(Point(0.20, 0.20), 0.05)
mesh = generate_mesh(Ω, 64)

# Function space and boundary markers
V   = df.FunctionSpace(mesh, "P", 1)
bmarks = df.MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
df.plot(bmarks, title="Boundary markers")
df.interact()

# Variational formulation
u, v = df.TrialFunction(V), df.TestFunction(V)
p, q = df.TrialFunction(df.FunctionSpace(mesh, "P", 1)), df.TestFunction(df.FunctionSpace(mesh, "P", 1))
mu, rho = df.Constant(0.001), df.Constant(1.0)

# Inlet
df.sub(0).sub(0).assign(df.Expression("6*x[1]*(H-x[1])/(H*H)", H=0.41, degree=2))

# Walls and circle
df.sub(0).sub(1).set_all(0)
df.sub(0).sub(2).set_all(0)

# Circular boundary
bmarks[3] = 1

# Variational form
a = mu * df.dot(df.grad(u), df.grad(v)) * df.dx - p * q * df.dx
L = rho * u * v * df.dx

# Solve
u_sol, p_sol = df.Function(V), df.Function(df.FunctionSpace(mesh, "P", 1))
df.solve(a == L, u_sol, bmarks)
df.solve(-p * df.grad(u_sol)[0] + df.div(p_sol) * u_sol == 0, p_sol)

# Compute drag coefficient
D = Circle(Point(0.20, 0.20), 0.05).area()
F_D = (rho * df.dot(df.grad(u_sol), df.Constant((0, -1))))[0] * df.ds

C_D = 2 * F_D / (rho * u_sol.vector().get_local() ** 2 * D)

# Save
df.File("q7_soln.xdmf") << u_sol
df.File("q7_soln.pvd")     << p_sol
df.plot(u_sol, title="Velocity |u|")
df.plot(p_sol, title="Pressure p")

# Save as PNG
import matplotlib.pyplot as plt
plt.savefig("q7_speed.png", dpi=300)