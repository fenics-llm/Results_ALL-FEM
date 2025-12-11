from fenics import *

# Define geometry parameters
a = 0.03  # radius of circular holes
Lx = 1.20  # length of rectangular strip
Ly = 0.20  # width of rectangular strip
xc = [0.30, 0.60, 0.90]  # x-coordinates of hole centers
yc = 0.10  # y-coordinate of hole center

# Define material parameters
lambda_ = 5.769e6  # Lame parameter (Pa)
mu = 3.846e6  # shear modulus (Pa)

# Create mesh
mesh = RectangleMesh(Point(0, 0), Point(Lx, Ly), 120, 20)

# Define function space
V = VectorFunctionSpace(mesh, 'P', 1)

# Define boundary conditions
def left_boundary(x, on_boundary):
    return near(x[0], 0) and on_boundary

def right_boundary(x, on_boundary):
    return near(x[0], Lx) and on_boundary

bc_left = DirichletBC(V, Constant((0, 0)), left_boundary)
bc_right = DirichletBC(V, Constant((0.012, 0)), right_boundary)

# Define variational problem
u = Function(V)
du = TrialFunction(V)
v = TestFunction(V)
F = Identity(2) + grad(u)
E = 0.5 * (F.T * F - Identity(2))
S = lambda_ * tr(E) * Identity(2) + 2 * mu * E
J = det(F)
sigma = (1 / J) * F * S * F.T

a_form = inner(grad(du), sigma) * dx
L_form = inner(Constant((0, 0)), v) * dx

# Define nonlinear problem
problem = NonlinearVariationalProblem(a_form - L_form, u, [bc_left, bc_right])
solver = NonlinearVariationalSolver(problem)

# Solve the nonlinear problem
solver.solve()

# Save plot of deformed configuration
plot(u, title='Deformed Configuration')
interactive()
import matplotlib.pyplot as plt
plt.savefig('q15_def.png')

# Compute and save color map of maximum principal value E_max
E_max_expr = Expression('sqrt(max((0.5 * (F[0][0] + F[1][1]))**2, (0.5 * (F[0][0] - F[1][1]))**2))', degree=0, F=F)
E_max_proj = project(E_max_expr, FunctionSpace(mesh, 'P', 1))
plot(E_max_proj, title='Maximum Principal Green-Lagrange Strain')
interactive()
plt.savefig('q15_Emax.png')

# Define and save color map of von Mises stress
s = S - (1 / 3) * tr(S) * Identity(2)
sigma_vm_expr = Expression('sqrt(1.5 * ((S[0][0] - (1/3) * tr(S))**2 + (S[1][1] - (1/3) * tr(S))**2 + (S[0][1]**2)))', degree=0, S=S)
sigma_vm_proj = project(sigma_vm_expr, FunctionSpace(mesh, 'P', 1))
plot(sigma_vm_proj, title='Von Mises Stress')
interactive()
plt.savefig('q15_vmS.png')

# Export final displacement and E_max in XDMF format
u_file = XDMFFile('q15_u.xdmf')
E_max_file = XDMFFile('q15_Emax.xdmf')
u_file.write(u, 0)
E_max_file.write(E_max_proj, 0)

TERMINATE