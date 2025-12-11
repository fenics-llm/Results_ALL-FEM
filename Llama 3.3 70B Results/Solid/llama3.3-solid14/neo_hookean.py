# filename: neo_hookean.py
from fenics import *
import numpy as np

# Define geometry parameters
Lx = 1.0
Ly = 0.20
a = 0.04
center1 = (0.40, 0.10)
center2 = (0.60, 0.10)

# Create mesh
mesh = RectangleMesh(Point(0, 0), Point(Lx, Ly), 100, 20)

# Define material parameters
E = 5e6  # Pa
nu = 0.49

# Define boundary conditions and loading
class LeftEdge(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0) and on_boundary

class RightEdge(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], Lx) and on_boundary

class Hole1(SubDomain):
    def inside(self, x, on_boundary):
        return ((x[0] - center1[0])**2 + (x[1] - center1[1])**2 < a**2) and on_boundary

class Hole2(SubDomain):
    def inside(self, x, on_boundary):
        return ((x[0] - center2[0])**2 + (x[1] - center2[1])**2 < a**2) and on_boundary

# Create boundary markers
boundaries = MeshFunction("size_t", mesh, 1)
left_edge = LeftEdge()
right_edge = RightEdge()
hole1 = Hole1()
hole2 = Hole2()

left_edge.mark(boundaries, 1)
right_edge.mark(boundaries, 2)
hole1.mark(boundaries, 3)
hole2.mark(boundaries, 4)

# Define function spaces
V = VectorFunctionSpace(mesh, "Lagrange", 2)
P = FunctionSpace(mesh, "Lagrange", 1)
W = MixedFunctionSpace([V, P])

# Define trial and test functions
u_p = TrialFunction(W)
du_p = TestFunction(W)

# Split trial and test functions into components
u, p = split(u_p)
v, q = split(du_p)

# Define material model (Neo-Hookean)
def neo_hookean(u):
    I = Identity(2)
    F = I + grad(u)
    C = F.T * F
    J = det(F)
    E = 0.5 * (C - I)
    sigma = (E / (1 + nu)) * inv(C) * (F * E * F.T)
    return sigma

# Define variational form
a = inner(neo_hookean(u), grad(v)) * dx - p * div(v) * dx - q * div(u) * dx

# Apply boundary conditions and loading
bc1 = DirichletBC(W.sub(0), Constant((0, 0)), boundaries, 1)
bc2 = DirichletBC(W.sub(0).sub(0), Constant(0.06), boundaries, 2)

P_hole = -100e3  # Pa

# Solve variational problem
u_p_ = Function(W)
solve(a == 0, u_p_, bcs=[bc1, bc2])

# Split solution into components
u_, p_ = u_p_.split()

# Compute von Mises stress
sigma_vm = sqrt(tr(neo_hookean(u_))**2 / 3)

# Save results to file
vtkfile_u = File("displacement.pvd")
vtkfile_p = File("pressure.pvd")
vtkfile_vm = File("von_mises_stress.pvd")

vtkfile_u << u_
vtkfile_p << p_
vtkfile_vm << sigma_vm

# Save color maps as PNG files
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plot(p_, title="Pressure")
plt.savefig("q14_p.png")

plt.subplot(1, 2, 2)
plot(sigma_vm, title="Von Mises Stress")
plt.savefig("q14_vm.png")

# Save displacement field in XDMF format
xdmffile_u = XDMFFile("displacement.xdmf")
xdmffile_u.write(u_, 0)

TERMINATE