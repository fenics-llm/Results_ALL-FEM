# filename: neo_hookean.py
from fenics import *
import numpy as np

# Geometry parameters
L = 1.0         # length
H = 0.20        # height
a = 0.04        # radius of circular hole
xc = 0.50       # x-coordinate of center of circular hole
yc = 0.10       # y-coordinate of center of circular hole

# Material parameters
E = 5e6         # Young's modulus (Pa)
nu = 0.5        # Poisson ratio

# Load parameters
P_hole = 1e5    # pressure on hole boundary (Pa)

# Create mesh
mesh = RectangleMesh(Point(0, 0), Point(L, H), 100, 20)

# Define function spaces
V = VectorFunctionSpace(mesh, 'P', 2)  # displacement space
Q = FunctionSpace(mesh, 'P', 1)        # pressure space

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

# Define strain and stress tensors
def epsilon(u):
    return sym(grad(u))

def sigma(u, p):
    I = Identity(2)
    F = I + grad(u)
    J = det(F)
    invF = inv(F)
    C = F.T * F
    Eneo = (1/2) * (C - I)
    Sneo = (E/(4*(1-nu**2))) * (tr(Eneo)*I + 2*Eneo)
    Pneo = J * invF * Sneo * invF.T
    return Pneo - p*I

# Define boundary conditions and loading
def left_boundary(x, on_boundary):
    return near(x[0], 0) and on_boundary

class HoleBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return (x[0] - xc)**2 + (x[1] - yc)**2 < a**2 + DOLFIN_EPS and on_boundary

hole = HoleBoundary()
bc = DirichletBC(V, Constant((0, 0)), left_boundary)

# Define variational problem
a_u = inner(sigma(u, p), epsilon(v))*dx
n = FacetNormal(mesh)
L_u = -P_hole * dot(v, n) * ds
a_p = div(u) * q * dx

# Solve system of equations
up = Function(V*Q)
F = a_u + a_p - L_u
bc_up = DirichletBC(V.sub(0), Constant(0), left_boundary)
solve(F == 0, up, bcs=[bc_up])

u_sol, p_sol = split(up)

# Save deformed configuration and von Mises stress
file = File("q13_def.pvd")
file << u_sol

von_mises = sqrt(3/2 * inner(dev(sigma(u_sol, p_sol)), dev(sigma(u_sol, p_sol))))
vm_file = File("q13_vm.pvd")
vm_file << project(von_mises, Q)

# Save displacement field in XDMF format
xdmf_file = XDMFFile("displacement.xdmf")
xdmf_file.write(u_sol)