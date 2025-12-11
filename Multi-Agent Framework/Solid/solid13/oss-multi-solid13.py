# -*- coding: utf-8 -*-
#
# Large deformation incompressible Neo-Hookean solid with a circular hole.
# Geometry: (0,1.0)×(0,0.20) m, hole radius a=0.04 m centred at (0.50,0.10).
# Material (plane strain): E = 5 MPa, ν = 0.5  →  μ = 1.666666e6 Pa.
# Boundary conditions:
#   left edge (x=0)      : u = (0,0) (Dirichlet)
#   hole boundary (r=a)  : follower pressure P_hole = 0.10 MPa (Neumann)
#   remaining edges       : traction free (natural)
# Output:
#   - magnified deformed mesh (×5) saved as q13_def.png
#   - von Mises stress saved as q13_vm.png
#   - displacement field saved in XDMF format (q13_disp.xdmf)
#
# -----------------------------------------------------------------
# NOTE: This script uses the legacy FEniCS (dolfin) interface.
# -----------------------------------------------------------------
from dolfin import *
from mshr import Rectangle, Circle, generate_mesh
import matplotlib
matplotlib.use('Agg')               # headless backend
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------
# 1. Mesh with facet markers
# -----------------------------------------------------------------
a = 0.04
domain = Rectangle(Point(0.0, 0.0), Point(1.0, 0.20)) - Circle(Point(0.50, 0.10), a)
mesh = generate_mesh(domain, 64)   # h≈0.01 m

# explicit volume measure (required for mixed forms)
dx = Measure('dx', domain=mesh)

# Mark boundaries
boundaries = MeshFunction('size_t', mesh, mesh.topology().dim() - 1, 0)

class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, DOLFIN_EPS)

class Hole(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and ((x[0] - 0.50)**2 + (x[1] - 0.10)**2 < (a + DOLFIN_EPS)**2)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 1.0, DOLFIN_EPS)

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.20, DOLFIN_EPS)

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0, DOLFIN_EPS)

Left().mark(boundaries, 1)
Hole().mark(boundaries, 2)
Right().mark(boundaries, 3)
Top().mark(boundaries, 4)
Bottom().mark(boundaries, 5)

ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# -----------------------------------------------------------------
# 2. Mixed Taylor–Hood space (u,p)
# -----------------------------------------------------------------
Ve = VectorElement('Lagrange', mesh.ufl_cell(), 2)
Pe = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
TH = MixedElement([Ve, Pe])
W  = FunctionSpace(mesh, TH)

# Trial / test functions
w = Function(W)               # (u,p)
(u, p) = split(w)
(v, q) = TestFunctions(W)

# -----------------------------------------------------------------
# 3. Material parameters
# -----------------------------------------------------------------
E  = 5.0e6               # Pa
nu = 0.5
mu = E / (2.0 * (1.0 + nu))   # shear modulus
P_hole = 0.10e6                # follower pressure (Pa)

# -----------------------------------------------------------------
# 4. Kinematics (using trial functions for the residual)
# -----------------------------------------------------------------
I = Identity(2)
F = I + grad(u)               # deformation gradient
J = det(F)
b = F * F.T                   # left Cauchy–Green tensor

# -----------------------------------------------------------------
# 5. Weak form (incompressible Neo-Hookean)
# -----------------------------------------------------------------
P = mu*F - p*inv(F).T
n = FacetNormal(mesh)
R = inner(P, grad(v)) * dx - q * (J - 1.0) * dx - P_hole*dot(J*inv(F).T*n, v) * ds(2)
Jac = derivative(R, w, TrialFunction(W))

# -----------------------------------------------------------------
# 6. Boundary conditions
# -----------------------------------------------------------------
bcs = [DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 1)]

# Pressure gauge to fix nullspace (pointwise at (0,0))
class PointGauge(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and near(x[1], 0.0)

point_gauge = PointGauge()
bc_p = DirichletBC(W.sub(1), Constant(0.0), point_gauge, method='pointwise')
bcs.append(bc_p)

# -----------------------------------------------------------------
# 7. Solve the nonlinear problem (Newton)
# -----------------------------------------------------------------
solve(R == 0, w, bcs,
      solver_parameters={'newton_solver':
                         {'relative_tolerance':1e-6,
                          'absolute_tolerance':1e-8,
                          'maximum_iterations':30,
                          'linear_solver':'mumps'}})

# Split solution
(u_sol, p_sol) = w.split(deepcopy=True)

# -----------------------------------------------------------------
# 8. Post-processing
# -----------------------------------------------------------------
# Re-build kinematics with the converged displacement
F = I + grad(u_sol)
J = det(F)
b = F * F.T

# Cauchy stress
sigma = -p_sol * I + mu * b

# von Mises stress (plane strain)
s_dev = sigma - (1.0/3.0) * tr(sigma) * I
sigma_vm = sqrt(3.0/2.0 * inner(s_dev, s_dev))

# Project von Mises onto a scalar space for plotting
V_scalar = FunctionSpace(mesh, 'P', 1)
sigma_vm_proj = project(sigma_vm, V_scalar)

# -----------------------------------------------------------------
# 9. Save results
# -----------------------------------------------------------------
# XDMF displacement
with XDMFFile(mesh.mpi_comm(), "q13_disp.xdmf") as xdmf:
    xdmf.write(u_sol, 0.0)

# Plot magnified deformed mesh
X = mesh.coordinates()
U = u_sol.compute_vertex_values(mesh).reshape((-1, 2))
X_def = X + 5.0 * U

plt.figure(figsize=(6, 3))
plt.triplot(X_def[:, 0], X_def[:, 1], mesh.cells(), linewidth=0.5, color='gray')
plt.axis('equal')
plt.title('Magnified deformed configuration (×5)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.tight_layout()
plt.savefig("q13_def.png", dpi=300)

# Plot von Mises stress
plt.figure(figsize=(6, 3))
c = plot(sigma_vm_proj, title='Von Mises stress (Pa)', cmap='viridis')
plt.colorbar(c)
plt.axis('equal')
plt.tight_layout()
plt.savefig("q13_vm.png", dpi=300)

print("Simulation completed successfully.")