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

mesh = generate_mesh(domain, 160)

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
    def __init__(self, center, tol=1e-3):
        super().__init__()
        self.c = center
        self.tol = tol
    def inside(self, x, on_boundary):
        if not on_boundary:
            return False
        r = np.sqrt((x[0] - self.c.x())**2 + (x[1] - self.c.y())**2)
        return near(r, a, self.tol)

left  = Left()
right = Right()
hole1 = Hole(c1, tol=1e-3)
hole2 = Hole(c2, tol=1e-3)

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

# Traction on right edge: (2 MPa * m, 0) == (2e6, 0) in 2D per-unit-thickness convention
t = Constant((2.0e6, 0.0))

a_form = inner(sigma(u), sym(grad(v)))*dx
L_form = dot(t, v)*ds(2)

bc = DirichletBC(V, Constant((0.0, 0.0)), left)

u_sol = Function(V)
solve(a_form == L_form, u_sol, bc, solver_parameters={"linear_solver": "mumps"})

# -------------------------------------------------
# 6. Post-processing: von Mises stress (project to a Function)
# -------------------------------------------------
s = sigma(u_sol)
sxx = s[0, 0]
syy = s[1, 1]
sxy = s[0, 1]

von_mises_expr = sqrt(sxx**2 - sxx*syy + syy**2 + 3.0*sxy**2)

Vsig = FunctionSpace(mesh, "Lagrange", 1)
von_mises = project(von_mises_expr, Vsig)

with XDMFFile(mesh.mpi_comm(), "plate_disp.xdmf") as xdmf:
    xdmf.write(mesh)
    xdmf.write(u_sol, 0.0)

plt.figure(figsize=(8, 3))
p = plot(von_mises/1.0e6, title="Von Mises stress (MPa)", cmap="viridis")
plt.colorbar(p, label="MPa")
plt.axis('off')
plt.savefig("q4_vm.png", dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------
# 7. Max stress on hole boundaries and Kt (ONLY on marked hole boundary vertices)
# -------------------------------------------------
# Ensure facet->vertex connectivity is built
tdim = mesh.topology().dim()
mesh.init(tdim-1, 0)

hole_vertex_ids = set()
for marker in (3, 4):
    for facet in SubsetIterator(boundaries, marker):
        for vid in facet.entities(0):
            hole_vertex_ids.add(int(vid))

# For CG1, vertex_to_dof_map maps vertex index -> dof index
v2d = vertex_to_dof_map(Vsig)
vm_vals = von_mises.vector().get_local()

sigma_max = max(vm_vals[v2d[vid]] for vid in hole_vertex_ids)
sigma_max_MPa = sigma_max / 1.0e6
Kt = sigma_max_MPa / 2.0

print("Maximum von Mises stress on hole boundaries: {:.2f} MPa".format(sigma_max_MPa))
print("Stress concentration factor Kt = {:.3f}".format(Kt))
