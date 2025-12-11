from fenics import *
import numpy as np

# Define geometry parameters
Lx, Ly = 1.0, 0.20
a = 0.04
xc1, yc1 = 0.33, 0.10
xc2, yc2 = 0.67, 0.10

# Create mesh
mesh = RectangleMesh(Point(0, 0), Point(Lx, Ly), 100, 20)

# Define subdomains for holes
class Hole1(SubDomain):
    def inside(self, x, on_boundary):
        return (x[0] - xc1)**2 + (x[1] - yc1)**2 < a**2

class Hole2(SubDomain):
    def inside(self, x, on_boundary):
        return (x[0] - xc2)**2 + (x[1] - yc2)**2 < a**2

# Mark subdomains
subdomains = MeshFunction("size_t", mesh, 2)
subdomains.set_all(0)
hole1 = Hole1()
hole1.mark(subdomains, 1)
hole2 = Hole2()
hole2.mark(subdomains, 2)

# Define material parameters
E = 200e3  # Young's modulus in MPa
nu = 0.30  # Poisson's ratio

# Define boundary conditions
u_L = Constant((0, 0))  # Clamped left edge
t_R = Constant((2, 0))  # Traction on right edge

# Define function space and test functions
V = VectorFunctionSpace(mesh, "Lagrange", 2)
u = TrialFunction(V)
v = TestFunction(V)

# Define stress tensor and strain tensor
def sigma(u):
    return E / (1 + nu) * (grad(u) + grad(u).T) + E * nu / ((1 + nu) * (1 - 2 * nu)) * tr(grad(u)) * Identity(2)

def epsilon(v):
    return sym(grad(v))

# Define variational problem
a = inner(sigma(u), epsilon(v)) * dx
L = dot(t_R, v) * ds

# Apply boundary condition
bc = DirichletBC(V, u_L, 'on_boundary && near(x[0], 0)')

# Assemble and solve system
A = assemble(a)
b = assemble(L)
bc.apply(A, b)
u_sol = Function(V)
solve(A, u_sol.vector(), b)

# Compute von Mises stress
von_mises_expr = sqrt(((sigma(u_sol)[0, 0] - sigma(u_sol)[1, 1])**2 + (sigma(u_sol)[1, 1] - 0)**2 + (0 - sigma(u_sol)[0, 0])**2) / 2 +
                 3 * ((sigma(u_sol)[0, 1]**2 + sigma(u_sol)[1, 0]**2)))
von_mises = project(von_mises_expr, FunctionSpace(mesh, 'P', 1))

# Save von Mises stress to file
vtkfile = File("q4_vm.pvd")
vtkfile << von_mises

# Save displacement field to XDMF format
xdmffile = XDMFFile("displacement.xdmf")
xdmffile.write(u_sol, 0)

# Compute maximum von Mises stress at hole boundary
max_von_mises = np.max(von_mises.compute_vertex_values(mesh))

# Compute Stress Concentration Factor
Kt = max_von_mises / 2

print("Maximum von Mises stress at hole boundary:", max_von_mises)
print("Stress Concentration Factor (Kt):", Kt)