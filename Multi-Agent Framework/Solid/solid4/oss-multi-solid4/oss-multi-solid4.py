# -*- coding: utf-8 -*-
#
# Plane-stress elasticity of a rectangular plate with two circular holes.
# Left edge clamped, right edge loaded with 2 MPa tension.
# Outputs: displacement (XDMF), von Mises stress (PNG), max stress on holes and Kt.
#
from dolfin import *
from mshr import Rectangle, Circle, generate_mesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------
# 1. Geometry and mesh
# -------------------------------------------------
L, H = 1.0, 0.20
a = 0.04
c1 = Point(0.33, 0.10)
c2 = Point(0.67, 0.10)

domain = Rectangle(Point(0.0, 0.0), Point(L, H)) \
         - Circle(c1, a, 64) \
         - Circle(c2, a, 64)

mesh = generate_mesh(domain, 80)

# -------------------------------------------------
# 2. Function space (quadratic vector)
# -------------------------------------------------
V = VectorFunctionSpace(mesh, "Lagrange", 2)

# -------------------------------------------------
# 3. Boundary markers
# -------------------------------------------------
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, DOLFIN_EPS)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], L, DOLFIN_EPS)

class Hole(SubDomain):
    def __init__(self, center):
        super().__init__()
        self.c = center
    def inside(self, x, on_boundary):
        return on_boundary and ((x[0]-self.c.x())**2 + (x[1]-self.c.y())**2 < (a+1e-6)**2)

left  = Left()
right = Right()
hole1 = Hole(c1)
hole2 = Hole(c2)

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
left.mark(boundaries, 1)
right.mark(boundaries, 2)
hole1.mark(boundaries, 3)
hole2.mark(boundaries, 4)

ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# -------------------------------------------------
# 4. Material parameters (plane stress)
# -------------------------------------------------
E, nu = 2.0e11, 0.30
mu = E / (2.0*(1.0 + nu))
lambda_ps = 2.0*mu*nu/(1.0 - nu)

def sigma(v):
    eps = sym(grad(v))
    return 2.0*mu*eps + lambda_ps*tr(eps)*Identity(2)

# -------------------------------------------------
# 5. Variational problem
# -------------------------------------------------
u = TrialFunction(V)
v = TestFunction(V)

t = Constant((2.0e6, 0.0))

a_form = inner(sigma(u), sym(grad(v)))*dx
L_form = dot(t, v)*ds(2)

bc = DirichletBC(V, Constant((0.0, 0.0)), left)

u_sol = Function(V)
solve(a_form == L_form, u_sol, bc, solver_parameters={"linear_solver": "mumps"})

# -------------------------------------------------
# 6. Post-processing: von Mises stress
# -------------------------------------------------
s = sigma(u_sol)
sxx = s[0, 0]
syy = s[1, 1]
sxy = s[0, 1]

von_mises = sqrt(sxx**2 - sxx*syy + syy**2 + 3.0*sxy**2)

with XDMFFile(mesh.mpi_comm(), "plate_disp.xdmf") as xdmf:
    xdmf.write(u_sol, 0.0)

plt.figure(figsize=(8, 3))
p = plot(von_mises/1.0e6, title="Von Mises stress (MPa)", cmap="viridis")
plt.colorbar(p, label="MPa")
plt.axis('off')
plt.savefig("q4_vm.png", dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------
# 7. Max stress on hole boundaries and Kt
# -------------------------------------------------
V0 = FunctionSpace(mesh, "Lagrange", 1)
von_mises_proj = project(von_mises, V0)

coords = V0.tabulate_dof_coordinates().reshape((-1, 2))
vals = von_mises_proj.vector().get_local()
r1 = np.sqrt((coords[:,0]-c1.x())**2 + (coords[:,1]-c1.y())**2)
r2 = np.sqrt((coords[:,0]-c2.x())**2 + (coords[:,1]-c2.y())**2)
h = mesh.hmin()
mask = (np.abs(r1 - a) < h) | (np.abs(r2 - a) < h)
sigma_max = vals[mask].max()
sigma_max_MPa = sigma_max / 1.0e6
Kt = sigma_max_MPa / 2.0

print("Maximum von Mises stress on hole boundaries: {:.2f} MPa".format(sigma_max_MPa))
print("Stress concentration factor Kt = {:.3f}".format(Kt))