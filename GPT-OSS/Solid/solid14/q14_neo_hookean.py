# filename: q14_neo_hookean.py
from dolfin import *
import mshr
import matplotlib.pyplot as plt
import math

# -------------------------------------------------
# 1. Geometry: rectangle (0,1)×(0,0.20) with two holes
# -------------------------------------------------
Lx, Ly = 1.0, 0.20
a = 0.04
hole1_center = Point(0.40, 0.10)
hole2_center = Point(0.60, 0.10)

rect  = mshr.Rectangle(Point(0.0, 0.0), Point(Lx, Ly))
hole1 = mshr.Circle(hole1_center, a, 64)
hole2 = mshr.Circle(hole2_center, a, 64)

domain = rect - hole1 - hole2
mesh   = mshr.generate_mesh(domain, 80)   # mesh resolution

# -------------------------------------------------
# 2. Function space (displacement only, P2)
# -------------------------------------------------
V = VectorFunctionSpace(mesh, "CG", 2)

# -------------------------------------------------
# 3. Boundary definitions
# -------------------------------------------------
tol = 1e-8

class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, tol)

class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], Lx, tol)

class HoleBoundary(SubDomain):
    def inside(self, x, on_boundary):
        r1 = math.hypot(x[0] - hole1_center.x(), x[1] - hole1_center.y())
        r2 = math.hypot(x[0] - hole2_center.x(), x[1] - hole2_center.y())
        return on_boundary and (near(r1, a, 1e-3) or near(r2, a, 1e-3))

left_bc   = LeftBoundary()
right_bc  = RightBoundary()
hole_bc   = HoleBoundary()

# Mark facets for Neumann (hole) traction
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
hole_bc.mark(boundaries, 1)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# Dirichlet BCs (displacement)
zero = Constant((0.0, 0.0))
bcs = [DirichletBC(V, zero, left_bc)]

right_disp = Constant((0.06, 0.0))
bcs.append(DirichletBC(V, right_disp, right_bc))

# -------------------------------------------------
# 4. Material parameters (Neo‑Hookean, nearly incompressible)
# -------------------------------------------------
E  = 5.0e6          # Pa
nu = 0.49
mu = E / (2.0 * (1.0 + nu))          # shear modulus
kappa = E / (3.0 * (1.0 - 2.0 * nu)) # bulk modulus (large)

# -------------------------------------------------
# 5. Variational formulation (displacement only)
# -------------------------------------------------
u = Function(V)            # unknown displacement
v = TestFunction(V)        # test function

d = u.geometric_dimension()
I = Identity(d)

# Deformation gradient and invariants
F = I + grad(u)
J = det(F)
C = F.T * F
I1 = tr(C)
I1_bar = J**(-2.0/3.0) * I1   # iso‑choric invariant

# Strain‑energy density (iso‑choric + volumetric)
psi = mu/2.0 * (I1_bar - d) + kappa/2.0 * (J - 1.0)**2

# Internal virtual work (derivative of ψ)
R_int = derivative(psi*dx, u, v)

# Follower pressure on hole boundaries (P_hole = 0.10 MPa)
P_hole = Constant(0.10e6)   # Pa
N0 = FacetNormal(mesh)
# Traction in reference configuration: t = -P_hole * J * F^{-T} * N0
t = -P_hole * J * inv(F).T * N0
R_Neumann = inner(t, v) * ds(1)

# Total residual
R = R_int + R_Neumann

# Jacobian
Jac = derivative(R, u, TrialFunction(V))

# -------------------------------------------------
# 6. Solve the nonlinear problem
# -------------------------------------------------
problem = NonlinearVariationalProblem(R, u, bcs, Jac)
solver  = NonlinearVariationalSolver(problem)

prm = solver.parameters
prm['newton_solver']['absolute_tolerance'] = 1e-8
prm['newton_solver']['relative_tolerance'] = 1e-7
prm['newton_solver']['maximum_iterations'] = 30
prm['newton_solver']['linear_solver'] = 'mumps'
prm['newton_solver']['relaxation_parameter'] = 1.0   # full Newton step

solver.solve()

# -------------------------------------------------
# 7. Post‑processing
# -------------------------------------------------
# Compute pressure field p = κ (J‑1)
J_val = det(I + grad(u))
p = project(kappa * (J_val - 1.0), FunctionSpace(mesh, "CG", 1))

# Cauchy stress σ = (μ/J)(B - ⅓ I1̄ I) + p I   (plane strain)
B = F * F.T
sigma_expr = (mu / J_val) * (B - (1.0/3.0) * I1_bar * I) + p * I
sigma = project(sigma_expr, TensorFunctionSpace(mesh, "CG", 1))

# von Mises stress (plane strain)
s_dev = sigma - (1.0/2.0) * tr(sigma) * I   # deviatoric part
von_mises = project(sqrt(3.0/2.0 * inner(s_dev, s_dev)), FunctionSpace(mesh, "CG", 1))

# -------------------------------------------------
# 8. Save results
# -------------------------------------------------
# Pressure map
plt.figure()
p_plot = plot(p, title='Hydrostatic pressure $p$ (Pa)', cmap='viridis')
plt.colorbar(p_plot)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.tight_layout()
plt.savefig('q14_p.png', dpi=300)

# von Mises map
plt.figure()
vm_plot = plot(von_mises, title='Von Mises stress (Pa)', cmap='viridis')
plt.colorbar(vm_plot)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.tight_layout()
plt.savefig('q14_vm.png', dpi=300)

# Displacement field (XDMF)
xdmf_file = XDMFFile(mesh.mpi_comm(), "q14_displacement.xdmf")
xdmf_file.parameters["functions_share_mesh"] = True
xdmf_file.parameters["flush_output"] = True
xdmf_file.write(u, 0.0)
xdmf_file.close()

print("Simulation completed.")
print("  - Pressure image saved as q14_p.png")
print("  - Von Mises image saved as q14_vm.png")
print("  - Displacement field saved as q14_displacement.xdmf")