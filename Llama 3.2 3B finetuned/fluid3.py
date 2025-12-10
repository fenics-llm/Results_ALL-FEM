from dolfin import *
import mshr

# Geometry
mesh = mshr.Rectangle(Point(0, 0), Point(1, 1)) - mshr.Circle(Point(0.5, 0.5), 0.25)
domain = Mesh("q3_mesh.xml", "q3_mesh.msh")

# Function space
V = VectorFunctionSpace(domain, "P", 2)

# Variational formulation
u, v = TrialFunction(V), TestFunction(V)
p, q = TestFunction(V), TrialFunction(V)
f = Constant((0.0, -9.81))
a = (inner(grad(u), grad(v)) + div(v)*p + q*div(u))*dx
L = dot(f, v)*dx

# Boundary conditions
u_top = Expression(("1.0", "0.0"), degree=2)
bcs = [DirichletBC(V.sub(0), u_top, "on_boundary")]

# Parameters
rho, mu = 1.0, 1.0

# Solve
u_soln, p_soln = Function(V), Function(V)
solve(a == L, (u_soln, p_soln), bcs=bcs)

# Save results
File("q3_speed.pvd") << u_soln
File("q3_pressure.pvd") << p_soln
