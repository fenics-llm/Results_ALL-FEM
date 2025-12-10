# filename: legacy_fenics.py

from fenics import *

# Define the problem domain
class Rectangle:
    def __init__(self, center, size):
        self.center = center
        self.size = size

    def get_bounds(self):
        return (self.center[0] - self.size[0]/2, self.center[0] + self.size[0]/2), \
               (self.center[1] - self.size[1]/2, self.center[1] + self.size[1]/2)

class UnitSquareMesh:
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny

# Define the problem domain
Omega = Rectangle((0.5, 0.1), (1.0, 0.20))

# Create a mesh of the domain
mesh = UnitSquareMesh(100, 20)

# Define the boundary conditions
def left_bc(x):
    return DirichletBC(V, Constant(0), x[0])

def right_bc(x):
    return DirichletBC(V, Constant((2e6, 0)), x[1])

def hole_bc(x):
    return DirichletBC(V, Constant((0, 0)), x[0] - 0.5)

bc_left = left_bc(Omega)
bc_right = right_bc(Omega)
bc_hole = hole_bc(Omega)

# Define the function space for the displacement field
V = FunctionSpace(mesh, 'P', 1)

# Create a linear elasticity problem
problem = LinearElasticProblem(E, nu)

# Solve the problem
u = problem.solve(bc_left, bc_right, bc_hole)

# Compute the von Mises equivalent stress field
sigma = problem.compute_stress(u)
vmis = problem.compute_vmis(sigma)

# Save the resulting displacement field in XDMF format
problem.save_xdmf('displacement.xdmf')

# Save a color map as q3_vm.png
import matplotlib.pyplot as plt

plt.imshow(vmis[0, :, :], cmap='viridis', origin='lower')
plt.colorbar()
plt.savefig('q3_vm.png')

# Report the maximum von Mises stress at the hole boundary and the Stress Concentration Factor (Kt = σ_max/2 MPa)
max_stress = problem.compute_max_stress(vmis[0, :, :])
print(f'Maximum von Mises stress: {max_stress} MPa')
print(f'Stress Concentration Factor (Kt): {max_stress / 2} MPa')
