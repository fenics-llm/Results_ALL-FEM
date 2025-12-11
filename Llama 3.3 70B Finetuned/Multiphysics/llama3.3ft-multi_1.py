from dolfin import *
import mshr
import numpy

# Geometry parameters
R0 = 0.05
s = 1e-4

# Time parameters
T_end = 10.0
dt = 0.01
num_steps = int(T_end / dt)

# Mesh generation
domain = mshr.Circle(Point(0, 0), R0)
mesh = mshr.generate_mesh(domain, 100)

# Function space
V = FunctionSpace(mesh, "CG", 1)

# Parameters
D = Constant(1e-5)  # diffusivity
kappa = Constant(1e-4)  # decay rate

# Mesh velocity
w_expr = Expression(("s*x[0]/sqrt(x[0]*x[0]+x[1]*x[1])", "s*x[1]/sqrt(x[0]*x[0]+x[1]*x[1])"),
                    s=s, degree=2)
w = as_vector(w_expr)

# Initial condition
c0 = interpolate(Expression("x[0]*x[0] + x[1]*x[1] < R*R ? 1.0 : 0.0", degree=2,
                            R=R0), V)
c_n = c0

# Variational formulation
u = TrialFunction(V)
v = TestFunction(V)

F = (1 / dt) * inner(u - c_n, v) * dx \
    + inner(D * grad(u), grad(v)) * dx \
    + kappa * u * v * dx \
    + inner(dot(w, grad(u)), v) * dx

a, L = lhs(F), rhs(F)

# Solver
c = Function(V)
problem = LinearVariationalProblem(a, L, c)
solver = LinearVariationalSolver(problem)

# Output file
file = XDMFFile("concentration.xdmf")
file.parameters["flush_output"] = True
file.parameters["rewrite_function_mesh"] = False

for n in range(num_steps):
    t = (n + 1) * dt
    
    # Move the mesh
    mesh.coordinates()[:] += dt * numpy.array([w_expr(x) for x in mesh.coordinates()])
    
    # Solve
    solver.solve()
    
    # Update
    c_n.assign(c)
    
    # Output
    file.write(c, t)
    
    if n % 100 == 0:
        print(f"Time {t:.2f}: Total concentration = {assemble(c * dx)}")