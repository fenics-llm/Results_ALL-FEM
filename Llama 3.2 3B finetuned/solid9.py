from dolfin import *
import mshr

# 1. Geometry: rectangular plate with height=0.20m
mesh = mshr.Rectangle(Point(0, 0), Point(1, 0.2)) + \
       mshr.Rectangle(Point(0, 0), Point(1, 0.8))
mesh = mesh.subdivide(100, 20)

# 2. Material: E(y) = 100 GPa + 100 GPa × (y / 0.20)
E = Expression("100.0 + 100.0 * x[1] / 0.2", degree=2)
nu = Constant(0.3)

# 3. Variational form: displacement u, stress σ
u, v = TrialFunction(V), TestFunction(V)
sigma = E*sym(grad(u))
a = inner(sigma, grad(v))*dx

# 4. Boundary conditions and loads
bc_left = DirichletBC(V, Constant((0.0, 0.0)), "near(x[0], 0.0)")
bcs = [bc_left]

# 5. Right edge traction: σ n = (2e6 Pa ⋅ m, 0)
def right_traction(x):
    return as_vector((2e6*x[1]/0.8, 0.0))

right_traction = Constant(right_traction)

# 6. Top and bottom: free
bcs.append(DirichletBC(V, Constant((0.0, 0.0)), "near(x[1], 0.2)"))

# 7. Solve displacement u
u_sol = Function(V)
solve(a == 0, u_sol, bcs)

# 8. Output: |u| (magnitude of displacement vector)
disp = project(sqrt(dot(u_sol, u_sol)), FunctionSpace(mesh, "P", 1))
plot(disp, title="Problem 9: Displacement magnitude")
interactive()

# 9. Save displacement field in XDMF format
File("q9_u.xdmf") << u_sol
