from dolfin import *
import mshr

# Geometry
mesh = mshr.Rectangle(Point(0, 0), Point(1, 0.2)) + \
       mshr.Rectangle(Point(0, 0), Point(1, 0))
mesh = mesh.subdivide(128)
domain = Mesh("q9_mesh.xml")

# Function space
V = VectorFunctionSpace(domain, "P", 1)

# Body force
G = Constant((1.0, 0.0))

# Variational problem
u, v = TrialFunction(V), TestFunction(V)
a = (mu*dot(grad(u), grad(v)) - div(v)*p) * dx
L = dot(G, v) * dx

# Boundary conditions
bc_y0 = DirichletBC(V.sub(0), Constant((0.0, 0.0)), "near(x[1], 0.0)")
bc_yH = DirichletBC(V.sub(0), Constant((0.0, 0.0)), "near(x[1], 0.2)")

# Solve
u_soln = Function(V)
solve(a == L, u_soln, [bc_y0, bc_yH])

# Save solution
File("q9_soln.pvd") << u_soln

