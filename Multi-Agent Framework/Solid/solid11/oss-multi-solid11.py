# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dolfin import *
import mshr

# ----------------------------------------------------------------------
# 1. Geometry & mesh
# ----------------------------------------------------------------------
Lx, Ly = 1.0, 0.20
a, xc, yc = 0.04, 0.50, 0.10

domain = mshr.Rectangle(Point(0.0, 0.0), Point(Lx, Ly)) \
         - mshr.Circle(Point(xc, yc), a, 64)
mesh = mshr.generate_mesh(domain, 64)

# ----------------------------------------------------------------------
# 2. Mixed (Taylor–Hood) function space
# ----------------------------------------------------------------------
Ve = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Pe = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = MixedElement([Ve, Pe])
W  = FunctionSpace(mesh, TH)

# ----------------------------------------------------------------------
# 3. Boundary conditions
# ----------------------------------------------------------------------
class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, DOLFIN_EPS)

class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], Lx, DOLFIN_EPS)

left  = LeftBoundary()
right = RightBoundary()

bc_left  = DirichletBC(W.sub(0), Constant((0.0, 0.0)), left)
bc_right = DirichletBC(W.sub(0), Constant((0.001, 0.0)), right)
bcs = [bc_left, bc_right]

# ----------------------------------------------------------------------
# 4. Material parameters (plane strain)
# ----------------------------------------------------------------------
E, nu = 5.0e6, 0.49
mu    = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
kappa = lmbda + 2.0*mu/3.0

# ----------------------------------------------------------------------
# 5. Variational formulation (mixed displacement–pressure)
# ----------------------------------------------------------------------
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

epsilon = sym(grad(u))
a = (2.0 * mu) * inner(epsilon, sym(grad(v))) * dx \
    - p * div(v) * dx \
    + q * div(u) * dx \
    + (1.0 / kappa) * p * q * dx
L = dot(Constant((0.0, 0.0)), v) * dx   # zero traction

# ----------------------------------------------------------------------
# 6. Solve
# ----------------------------------------------------------------------
w = Function(W)
solve(a == L, w, bcs)

(u, p) = w.split(deepcopy=True)

# ----------------------------------------------------------------------
# 7. Post-processing: von Mises stress and ux
# ----------------------------------------------------------------------
sigma = 2.0 * mu * sym(grad(u)) - p * Identity(2)
s = sigma - (1.0 / 3.0) * tr(sigma) * Identity(2)
vonMises = sqrt(3.0 / 2.0 * inner(s, s))

Vproj = FunctionSpace(mesh, "P", 1)
vm_proj = project(vonMises, Vproj)
ux_proj = project(u[0], Vproj)

# ----------------------------------------------------------------------
# 8. PNG output
# ----------------------------------------------------------------------
plt.tripcolor(mesh.coordinates()[:, 0],
              mesh.coordinates()[:, 1],
              mesh.cells(),
              vm_proj.compute_vertex_values(mesh),
              shading='flat')
plt.colorbar()
plt.title('Von Mises stress (Pa)')
plt.savefig('q11_vm.png')
plt.clf()

plt.tripcolor(mesh.coordinates()[:, 0],
              mesh.coordinates()[:, 1],
              mesh.cells(),
              ux_proj.compute_vertex_values(mesh),
              shading='flat')
plt.colorbar()
plt.title('Horizontal displacement u_x (m)')
plt.savefig('q11_ux.png')
plt.clf()

# ----------------------------------------------------------------------
# 9. XDMF output (displacement + pressure)
# ----------------------------------------------------------------------
with XDMFFile(mesh.mpi_comm(), "q11_disp.xdmf") as xdmf:
    xdmf.write(u, 0.0)
    xdmf.write(p, 0.0)