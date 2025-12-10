from dolfin import *
import mshr

# 1. Geometry
mesh = mshr.Rectangle(Point(0, 0), Point(1, 0.2)) - \
       mshr.Circle(Point(0.4, 0.1), 0.04) - \
       mshr.Circle(Point(0.6, 0.1), 0.04)
mesh = generate_mesh(mesh, 64)

# 2. Material
E, nu = 5e3, 0.49
mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

# 3. Variational problem
V = VectorFunctionSpace(mesh, "P", 1)
u, v = TrialFunction(V), TestFunction(V)
p, q = Function(V), Function(V)

def sigma(u):
    return lmbda * tr(sym(grad(u))) * Identity(2) + 2 * mu * sym(grad(u))

a = inner(sigma(u), grad(v)) * dx
L = dot(grad(p), v) * dx

bc_left = DirichletBC(V, Constant((0.0, 0.0)), "near(x[0], 0.0)")
bc_right = DirichletBC(V, Constant((0.06, 0.0)), "near(x[0], 1.0)")

P_hole = Constant(10e3)
bcs = [bc_left, bc_right]

# 4. Solve
p_sol = Function(V)
solve(a == L, p_sol, bcs)

# 5. Output
mesh = mesh
plot(mesh)
plt.colorbar(plot(mesh))
plt.savefig("q14_mesh.png")

# 6. Von Mises stress
vm = sqrt(0.5 * (sigma(u) + sigma(u).T)[0][0] +
          0.5 * (sigma(u) + sigma(u).T)[1][1] -
          0.25 * ((sigma(u) + sigma(u).T)[0][0] +
                  (sigma(u) + sigma(u).T)[1][1]) ** 2)
vm_sol = Function(V)

solve(a == L, vm_sol, bcs)

# 7. Save
plot(vm_sol)
plt.colorbar(plot(vm_sol))
plt.savefig("q14_vm.png")