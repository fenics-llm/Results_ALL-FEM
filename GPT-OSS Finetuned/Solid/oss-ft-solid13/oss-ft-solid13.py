# filename: q13_neo_hookean.py
from dolfin import *
import mshr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # headless backend

# -------------------------------------------------
# Geometry and mesh
# -------------------------------------------------
L, H = 1.0, 0.20
a = 0.04
center = Point(0.5, 0.10)

rect = mshr.Rectangle(Point(0.0, 0.0), Point(L, H))
hole = mshr.Circle(center, a, 64)
domain = rect - hole
mesh = mshr.generate_mesh(domain, 80)   # mesh resolution

# -------------------------------------------------
# Mixed Taylor–Hood space (u,p)
# -------------------------------------------------
Ve = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Pe = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = MixedElement([Ve, Pe])
W  = FunctionSpace(mesh, TH)

# -------------------------------------------------
# Boundary markers
# -------------------------------------------------
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, DOLFIN_EPS)

class Hole(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and ((x[0]-center.x())**2 + (x[1]-center.y())**2 < (a+1e-6)**2)

left = Left()
hole = Hole()

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
left.mark(boundaries, 1)
hole.mark(boundaries, 2)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# -------------------------------------------------
# Material parameters (plane strain, incompressible)
# -------------------------------------------------
E  = 5e6          # Pa
nu = 0.5
mu = E/(2.0*(1.0+nu))   # shear modulus

# -------------------------------------------------
# Mixed function, test and trial
# -------------------------------------------------
w = Function(W)               # (u,p)
(u, p) = split(w)
(v, q) = TestFunctions(W)

# -------------------------------------------------
# Kinematics
# -------------------------------------------------
d = u.geometric_dimension()
I = Identity(d)
F = I + grad(u)               # deformation gradient
J = det(F)
C = F.T*F
I1_bar = J**(-2.0/3.0)*tr(C)

# -------------------------------------------------
# Strain energy (incompressible Neo-Hookean)
# -------------------------------------------------
psi = mu/2.0*(I1_bar - 2.0) - p*(J - 1.0)
Pi  = psi*dx

# -------------------------------------------------
# Residual and Jacobian (mixed formulation)
# -------------------------------------------------
R = derivative(Pi, w, TestFunction(W))
Jform = derivative(R, w, TrialFunction(W))

# -------------------------------------------------
# Boundary conditions
# -------------------------------------------------
bcs = []

# Left edge: u = (0,0)
zero = Constant((0.0, 0.0))
bcs.append(DirichletBC(W.sub(0), zero, left))

# Pressure gauge (fix p at a point to remove nullspace)
class PointGauge(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and near(x[1], 0.0)
gauge = PointGauge()
bcs.append(DirichletBC(W.sub(1), Constant(0.0), gauge, method="pointwise"))

# -------------------------------------------------
# Follower pressure on hole boundary
# -------------------------------------------------
P_hole = 0.10e6   # Pa
n = FacetNormal(mesh)
R += -P_hole*dot(v, J*inv(F.T)*n)*ds(2)   # traction contribution

# -------------------------------------------------
# Newton solver
# -------------------------------------------------
dw = Function(W)   # Newton increment
tol = 1e-6
max_iter = 30
for iter in range(max_iter):
    R_vec = assemble(R)
    for bc in bcs:
        bc.apply(R_vec)
    J_mat = assemble(Jform)
    for bc in bcs:
        bc.apply(J_mat)
    solve(J_mat, dw.vector(), -R_vec, "lu")
    w.vector().axpy(1.0, dw.vector())
    residual = norm(R_vec, 'l2')
    print("Newton iteration {}: residual = {:.3e}".format(iter+1, residual))
    if residual < tol:
        break

(u_h, p_h) = w.split(deepcopy=True)

# -------------------------------------------------
# Cauchy stress and von Mises
# -------------------------------------------------
Vsig = TensorFunctionSpace(mesh, "DG", 0)
F_h = I + grad(u_h)
J_h = det(F_h)
B_h = F_h*F_h.T
I1_bar_h = J_h**(-2.0/3.0)*tr(B_h)

sigma_expr = -p_h*I + mu*J_h**(-2.0/3.0)*(B_h - (1.0/3.0)*I1_bar_h*I)
sigma = project(sigma_expr, Vsig)

s = sigma - (1.0/3.0)*tr(sigma)*I
vonMises = project(sqrt(3.0/2.0*inner(s, s)), FunctionSpace(mesh, "DG", 0))

# -------------------------------------------------
# Save displacement (XDMF)
# -------------------------------------------------
xdmf = XDMFFile(mesh.mpi_comm(), "q13_disp.xdmf")
xdmf.write(u_h)
xdmf.close()

# -------------------------------------------------
# Plot deformed mesh (magnified 5×) and save
# -------------------------------------------------
plt.figure(figsize=(8,4))
V1 = VectorFunctionSpace(mesh, "CG", 1)
u1 = interpolate(u_h, V1)
mesh_def = Mesh(mesh)
coords = mesh_def.coordinates()
nv = mesh_def.num_vertices()
vals = u1.compute_vertex_values(mesh_def)
coords[:, 0] += 5.0*vals[:nv]
coords[:, 1] += 5.0*vals[nv:]
plot(mesh_def)
plt.axis('off')
plt.title('Deformed configuration (×5)')
plt.savefig("q13_def.png", dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------
# Plot von Mises stress and save
# -------------------------------------------------
plt.figure(figsize=(6,3))
c = plot(vonMises, title='von Mises stress (Pa)', cmap='viridis')
plt.colorbar(c)
plt.axis('off')
plt.savefig("q13_vm.png", dpi=300, bbox_inches='tight')
plt.close()

print("Simulation completed. Files saved: q13_def.png, q13_vm.png, q13_disp.xdmf")