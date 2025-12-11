# -*- coding: utf-8 -*-
#
#  q15 - Finite strain Saint-Venant–Kirchhoff strip with three holes
#  (legacy dolfin, Python 3)
#
#  Geometry: 0 ≤ x ≤ 1.20, 0 ≤ y ≤ 0.20  with three circular holes
#  radius a = 0.03 centred at (0.30,0.10), (0.60,0.10), (0.90,0.10)
#
#  Material (plane strain):
#      λ = 5.769 MPa,   μ = 3.846 MPa
#
#  Loading: left edge fixed, right edge prescribed u = (α·0.012,0)
#           α increased stepwise until max principal Green-Lagrange strain
#           exceeds 0.03 (then stop before exceeding).
#
#  Outputs: deformed shape, E_max field, von-Mises stress based on S,
#           XDMF files for u and E_max.
#
# --------------------------------------------------------------
from dolfin import *
import mshr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# 1. Mesh with three holes
a = 0.03
Lx, Ly = 1.20, 0.20
domain = mshr.Rectangle(Point(0.0, 0.0), Point(Lx, Ly))
centers = [Point(0.30, 0.10), Point(0.60, 0.10), Point(0.90, 0.10)]
for c in centers:
    domain = domain - mshr.Circle(c, a, 64)
mesh = mshr.generate_mesh(domain, 120)

# --------------------------------------------------------------
# 2. Function space (Taylor-Hood: P2 vector)
V = VectorFunctionSpace(mesh, "Lagrange", 2)

# --------------------------------------------------------------
# 3. Boundary markers
tol = 1e-8
class Left(SubDomain):
    def inside(self, x, on):
        return on and near(x[0], 0.0, tol)
class Right(SubDomain):
    def inside(self, x, on):
        return on and near(x[0], Lx, tol)
left = Left()
right = Right()

# --------------------------------------------------------------
# 4. Dirichlet BCs
zero = Constant((0.0, 0.0))
bcs_fixed = [DirichletBC(V, zero, left)]
u_R = Expression(("alpha*0.012", "0.0"), alpha=0.0, degree=1)
bc_right = DirichletBC(V, u_R, right)

# --------------------------------------------------------------
# 5. Material parameters (MPa → Pa)
MPa = 1e6
lmbda = 5.769 * MPa
mu    = 3.846 * MPa

# --------------------------------------------------------------
# 6. Kinematics & energy
u = Function(V, name="Displacement")
v = TestFunction(V)
du = TrialFunction(V)

I = Identity(2)
F = I + grad(u)
E = 0.5*(F.T*F - I)
S = lmbda*tr(E)*I + 2.0*mu*E
P = F*S

R = inner(P, grad(v))*dx
Jac = derivative(R, u, du)

# --------------------------------------------------------------
# 7. Load stepping with strain limit
u_prev = Function(V)
alpha = 0.0
dalpha = 0.05
max_Emax = 0.0

while True:
    u_prev.assign(u)
    alpha_try = min(alpha + dalpha, 1.0)
    if alpha_try == alpha:
        break
    u_R.alpha = alpha_try
    bcs = bcs_fixed + [bc_right]
    solve(R == 0, u, bcs,
          solver_parameters={"newton_solver":
                             {"relative_tolerance": 1e-8,
                              "absolute_tolerance": 1e-8,
                              "maximum_iterations": 25,
                              "linear_solver": "mumps"}})
    E_proj = project(E, TensorFunctionSpace(mesh, "DG", 0))
    e_vals = np.array(E_proj.vector().get_local()).reshape((-1, 2, 2))
    eigs = np.linalg.eigvals(e_vals)
    principal = eigs.real.max(axis=1)
    max_Emax = float(principal.max())
    print("alpha_try = {:.3f},  max principal E = {:.5e}".format(alpha_try, max_Emax))
    if max_Emax > 0.03:
        u.assign(u_prev)
        break
    alpha = alpha_try

# --------------------------------------------------------------
# 9. Post-processing
mesh_def = Mesh(mesh)
ALE.move(mesh_def, u)

plt.figure(figsize=(8, 3))
plot(mesh_def, linewidth=0.3)
plt.title("Deformed configuration (α = {:.2f})".format(alpha))
plt.axis('off')
plt.savefig("q15_def.png", dpi=300)

# --------------------------------------------------------------
# 10. Max principal strain field
E_proj = project(E, TensorFunctionSpace(mesh, "DG", 0))
e_vals = np.array(E_proj.vector().get_local()).reshape((-1, 2, 2))
eigs = np.linalg.eigvals(e_vals)
principal = eigs.real.max(axis=1)
Emax_func = Function(FunctionSpace(mesh, "DG", 0))
Emax_func.vector().set_local(principal)
Emax_func.vector().apply("insert")

plt.figure()
c = plot(Emax_func, cmap='viridis')
plt.colorbar(c)
plt.title(r"$E_{\max}$ (Green-Lagrange)")
plt.savefig("q15_Emax.png", dpi=300)

# --------------------------------------------------------------
# 11. von Mises stress based on S
S_proj = project(S, TensorFunctionSpace(mesh, "DG", 0))
S11, S12, S22 = S_proj[0,0], S_proj[0,1], S_proj[1,1]
E11, E22 = E_proj[0,0], E_proj[1,1]
S33 = lmbda*(E11 + E22)
S3 = as_tensor([[S11, S12, 0.0],
                [S12, S22, 0.0],
                [0.0,  0.0,  S33]])
I3 = Identity(3)
s3 = S3 - (tr(S3)/3.0)*I3
vm = sqrt(1.5*inner(s3, s3))
vm_func = project(vm, FunctionSpace(mesh, "DG", 0))

plt.figure()
c = plot(vm_func, cmap='plasma')
plt.colorbar(c)
plt.title(r"$\sigma_{\text{vm}}(S)$")
plt.savefig("q15_vmS.png", dpi=300)

# --------------------------------------------------------------
# 12. XDMF export
with XDMFFile(mesh.mpi_comm(), "q15_u.xdmf") as xdmf:
    xdmf.write(u)

with XDMFFile(mesh.mpi_comm(), "q15_Emax.xdmf") as xdmf:
    xdmf.write(Emax_func)

print("Simulation finished. Results saved.")