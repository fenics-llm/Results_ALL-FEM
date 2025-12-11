from dolfin import *
from mshr import Rectangle, generate_mesh
import numpy as np
import matplotlib.pyplot as plt

# Geometry
mesh = generate_mesh(Rectangle(0, 0, 1.0, 0.2, 64), 64)
V = VectorFunctionSpace(mesh, "P", 1)

# Material (rotated orthotropic plane-stress)
E1, E2, G12, nu12 = 40e9, 10e9, 5e9, 0.10
theta = np.radians(30)
c, s = np.cos(theta), np.sin(theta)
Q = as_matrix([[E1/(1 - nu12**2), E1*nu12/(1 - nu12**2),
               E1*nu12/(1 - nu12**2), E2/(1 - nu12**2)])

R = as_matrix([[c**2, s**2],
               [s**2, c**2]])
Q_rot = R * Q * R.T

# Variational form
def eps(w): return sym(grad(w))
u, v = TrialFunction(V), TestFunction(V)
a = inner(Q_rot * eps(u), eps(v)) * dx
L = Constant((0.0, -10e6)) * v * ds(1)

# Boundary conditions
bdr = MeshFunction("size_t", mesh, 0)
class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0)
Bottom().mark(bdr, 1)
bc = DirichletBC(V, Constant((0.0, 0.0)), bdr, 1)

# Solve
solve(a == L, u, [bc])

# Post-process
vm = project(sqrt((Q_rot * eps(u))[0,0]**2 + (Q_rot * eps(u))[1,1]**2 - (Q_rot * eps(u))[0,1]**2 + 3*(Q_rot * eps(u))[0,1]**2, FunctionSpace(mesh, "P", 1))

File("q8_solution.pvd").write(u)
File("q8_solution.xdmf").write(u)
plt.figure(); plot(u.sub(0)); plt.colorbar(); plt.title("u_x"); plt.show()
plt.figure(); plot(vm); plt.colorbar(); plt.title("VM"); plt.show()