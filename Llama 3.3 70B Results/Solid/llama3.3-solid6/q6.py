from fenics import *
import numpy as np
import matplotlib.pyplot as plt

# Define geometry parameters
L = 1.0         # length
H = 0.20        # height
a = 0.05        # radius of semicircular notch
center_x = 0.5  # x-coordinate of center of semicircular notch
center_y = 0.2  # y-coordinate of center of semicircular notch

# Create mesh
mesh = RectangleMesh(Point(0, 0), Point(L, H), 100, 20)

# Define function space
V = VectorFunctionSpace(mesh, 'P', 1)

# Define boundary conditions
def bottom(x, on_boundary):
    return near(x[1], 0) and on_boundary

def top(x, on_boundary):
    return near(x[1], H) and (x[0] < 0.45 or x[0] > 0.55) and on_boundary

bc_bottom = DirichletBC(V, Constant((0, 0)), bottom)

# Define material parameters
E = 200e3  # Young's modulus in Pa
nu = 0.30   # Poisson's ratio

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant((0, -10e6))  # traction force on top edge

# Define stress and strain tensors
def epsilon(u):
    return sym(grad(u))

def sigma(u, E, nu):
    mu = E / (2 * (1 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    return 2 * mu * epsilon(u) + lmbda * tr(epsilon(u)) * Identity(2)

a_form = inner(sigma(u, E, nu), epsilon(v))*dx
L_form = dot(f, v)*ds

# Compute solution
u_sol = Function(V)
solve(a_form == L_form, u_sol, [bc_bottom])

# Compute von Mises stress
def von_mises(stress):
    return sqrt((stress[0, 0] - stress[1, 1])**2 + 3 * (stress[0, 1]**2 + stress[1, 0]**2) + 3 * (stress[0, 0]**2 + stress[1, 1]**2))

vm_stress = von_mises(sigma(u_sol, E, nu))
vm_stress_expr = project(vm_stress, FunctionSpace(mesh, 'P', 1))

# Save results
File("q6_vm.pvd") << vm_stress_expr
XDMFFile("displacement.xdmf").write(u_sol)
plot(vm_stress_expr, title="Von Mises Stress")
plt.savefig('q6_vm.png')