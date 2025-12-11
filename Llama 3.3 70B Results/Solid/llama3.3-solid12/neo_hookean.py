# filename: neo_hookean.py
from fenics import *
import numpy as np

# Geometry parameters
L = 1.0         # length
H = 0.20        # height
a = 0.04        # radius of circular hole
center_x, center_y = 0.50, 0.10  # center of circular hole

# Material properties (plane strain)
E = 5e6         # Young's modulus in Pa
nu = 0.5        # Poisson ratio

# Mesh generation
mesh = RectangleMesh(Point(0, 0), Point(L, H), 100, 20)

# Function spaces
V = VectorFunctionSpace(mesh, 'P', 1)  # displacement space
Q = FunctionSpace(mesh, 'P', 1)       # pressure space

# Trial and test functions
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

# Incompressible neo-Hookean strain-energy density (mixed formulation)
I = Identity(2)         # identity tensor
F = I + grad(u)         # deformation gradient
C = F.T * F             # right Cauchy-Green tensor
J = det(F)
psi = (E / (4 * (1 + nu))) * (tr(C) - 3)

# Variational formulation
a_u = inner(grad(v), F) * dx 
L_u = Constant(0) * v[0] * dx

a_p = p * q * dx
L_p = div(u) * q * dx

# Boundary conditions and loading
def left_boundary(x, on_boundary):
    return near(x[0], 0)

def right_boundary(x, on_boundary):
    return near(x[0], L)

bc_u_left = DirichletBC(V, Constant((0, 0)), left_boundary)
bc_u_right_x = DirichletBC(V.sub(0), Constant(0.060), right_boundary)
bc_u_right_y = DirichletBC(V.sub(1), Constant(0), right_boundary)
bcs_u = [bc_u_left, bc_u_right_x, bc_u_right_y]

# Solve the system of equations
u_sol = Function(V)
p_sol = Function(Q)

solve(a_u == L_u, u_sol, bcs=bcs_u)
solve(a_p == L_p, p_sol)

# Compute von Mises stress (from Cauchy stress)
F_inv = inv(F)
sigma = 2 * E / (4 * (1 + nu)) * F * (C - I) * F_inv.T
von_mises_stress = sqrt(3/2 * inner(dev(sigma), dev(sigma)))

# Save results to file
vtkfile_u = File('displacement.pvd')
vtkfile_p = File('pressure.pvd')
vtkfile_vm = File('von_mises_stress.pvd')

vtkfile_u << u_sol
vtkfile_p << p_sol

# Convert von Mises stress to a Function and save it
vm_func = project(von_mises_stress, Q)
vtkfile_vm << vm_func

# Save color maps as PNG files
import matplotlib.pyplot as plt

p_img = plot(p_sol, title='Hydrostatic Pressure')
plt.savefig('q12_p.png')

vm_img = plot(vm_func, title='Von Mises Stress')
plt.savefig('q12_vm.png')

# Save displacement field in XDMF format
xdmffile_u = XDMFFile('displacement.xdmf')
xdmffile_u.write(u_sol)