# filename: elasticity.py
from fenics import *

# Define geometry
Lx, Ly = 1.0, 0.20
nx, ny = 100, 20
mesh = RectangleMesh(Point(0, 0), Point(Lx, Ly), nx, ny)

# Define material parameters
E = 200e3  # Young's modulus in Pa
nu = 0.30  # Poisson's ratio

# Define boundary conditions
u_L = Constant((0, 0))  # displacement on left edge
t_R = Constant((2e6, 0))  # traction on right edge

# Define function space
V = VectorFunctionSpace(mesh, 'P', 1)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)

# Define stress and strain tensors
def epsilon(u):
    return sym(grad(u))

def sigma(u, E, nu):
    mu = E / (2 * (1 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    return 2 * mu * epsilon(u) + lmbda * tr(epsilon(u)) * Identity(2)

a = inner(sigma(u, E, nu), epsilon(v))*dx
L = dot(t_R, v)*ds(4)  # right edge has id 4

# Apply boundary conditions
bc_L = DirichletBC(V, u_L, 'on_boundary && near(x[0], 0)')

# Compute solution
u_sol = Function(V)
solve(a == L, u_sol, bcs=bc_L)

# Compute von Mises stress
def von_mises(sigma):
    return sqrt(3/2*inner(dev(sigma), dev(sigma)))

vm_stress = von_mises(sigma(u_sol, E, nu))

# Save results
file_vm = File('q5_vm.pvd')
file_vm << project(vm_stress, FunctionSpace(mesh, 'P', 1))

file_u = XDMFFile('u.xdmf')
file_u.write(u_sol, 0)
print("Solution computed and saved to file.")