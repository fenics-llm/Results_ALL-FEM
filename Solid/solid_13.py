# filename: q13_neo_hookean.py
"""
Large‑deformation incompressible Neo‑Hookean strip with a circular hole.

Outputs
-------
q13_def.png   : magnified (×5) deformed configuration
q13_vm.png    : von Mises stress colour map on the deformed mesh
displacement.xdmf : displacement field (XDMF)
"""

from dolfin import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------
# Geometry
Lx, Ly = 1.0, 0.20
a = 0.04
hole_center = Point(0.5, 0.10)

domain = Rectangle(Point(0.0, 0.0), Point(Lx, Ly)) - Circle(hole_center, a, 64)
mesh = generate_mesh(domain, 64)

# --------------------------------------------------------------------
# Mixed function space (displacement u, pressure p)
V_el = VectorElement("CG", mesh.ufl_cell(), 2)   # displacement (quadratic)
Q_el = FiniteElement("CG", mesh.ufl_cell(), 1)   # pressure (linear)
W_el = MixedElement([V_el, Q_el])
W = FunctionSpace(mesh, W_el)

# --------------------------------------------------------------------
# Boundary markers
left = CompiledSubDomain("near(x[0], side) && on_boundary", side=0.0)
hole = CompiledSubDomain(
    "pow(x[0]-cx,2)+pow(x[1]-cy,2) < r2+1e-8 && on_boundary",
    cx=hole_center.x(), cy=hole_center.y(), r2=a*a)

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
left.mark(boundaries, 1)
hole.mark(boundaries, 2)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

# --------------------------------------------------------------------
# Material parameters (plane strain)
E  = 5.0e6          # Pa
nu = 0.5
mu = E / (2.0 * (1.0 + nu))   # shear modulus
d  = 2                        # spatial dimension (plane strain)

# --------------------------------------------------------------------
# Unknown and test functions (mixed)
w = Function(W)                 # (u,p) unknown
(u, p) = split(w)               # for variational forms
(v, q) = TestFunctions(W)       # (v,q) test functions

# --------------------------------------------------------------------
# Kinematics
I = Identity(d)
F = I + grad(u)          # deformation gradient
J = det(F)
C = F.T * F
I1 = tr(C)

# Iso‑choric invariant
I1_bar = J**(-2.0/3.0) * I1

# Strain‑energy density (incompressible Neo‑Hookean)
psi = mu/2.0 * (I1_bar - d) - p * (J - 1.0)

# --------------------------------------------------------------------
# Follower pressure on the hole (P_hole = 0.10 MPa)
P_hole = 0.10e6          # Pa
N = FacetNormal(mesh)   # outward normal in reference configuration

# Current outward normal: n = J * F^{-T} * N
n_current = J * inv(F).T * N

# Virtual work of follower pressure (traction = -P_hole * n_current)
traction_term = -P_hole * dot(n_current, v) * ds(2)

# Total potential energy
Pi = psi * dx + traction_term

# --------------------------------------------------------------------
# Residual and Jacobian (mixed formulation)
R   = derivative(Pi, w, TestFunction(W))   # residual
Jac = derivative(R, w, TrialFunction(W))   # Jacobian

# --------------------------------------------------------------------
# Boundary conditions
zero_vec = Constant((0.0, 0.0))
bc_left = DirichletBC(W.sub(0), zero_vec, left)   # u = 0 on left edge
bcs = [bc_left]

# --------------------------------------------------------------------
# Solve the nonlinear problem
problem = NonlinearVariationalProblem(R, w, bcs, J=Jac)
solver = NonlinearVariationalSolver(problem)

prm = solver.parameters
prm["newton_solver"]["absolute_tolerance"] = 1e-8
prm["newton_solver"]["relative_tolerance"] = 1e-7
prm["newton_solver"]["maximum_iterations"] = 25
prm["newton_solver"]["linear_solver"] = "mumps"

solver.solve()

# --------------------------------------------------------------------
# Extract displacement and pressure
(u_sol, p_sol) = w.split()
u_sol.rename("displacement", "u")
p_sol.rename("pressure", "p")

# --------------------------------------------------------------------
# Post‑processing: Cauchy stress and von Mises stress
F_sol = I + grad(u_sol)
J_sol = det(F_sol)
B_sol = F_sol * F_sol.T
sigma_sol = mu / J_sol * (B_sol - I) - p_sol * I          # Cauchy stress

# Deviatoric part and von Mises stress
s_dev = sigma_sol - (1.0/3.0) * tr(sigma_sol) * I
von_mises = sqrt(3.0/2.0 * inner(s_dev, s_dev))

# Project to spaces suitable for plotting
Vsig = TensorFunctionSpace(mesh, "CG", 1)
Vvm  = FunctionSpace(mesh, "CG", 1)

sigma_proj = project(sigma_sol, Vsig)
vm_proj    = project(von_mises, Vvm)

# --------------------------------------------------------------------
# Save displacement field (XDMF)
xdmf = XDMFFile(mesh.mpi_comm(), "displacement.xdmf")
xdmf.write(u_sol)
xdmf.close()

# --------------------------------------------------------------------
# Plot deformed configuration (magnified ×5)
mesh_def = Mesh(mesh)                     # deep copy of the mesh
coords = mesh_def.coordinates()
u_vert = u_sol.compute_vertex_values().reshape((-1, d))
coords[:] = mesh.coordinates() + 5.0 * u_vert   # magnify displacement

plt.figure(figsize=(8, 4))
plot(mesh_def, linewidth=0.5, color="lightgray")
plt.title("Deformed configuration (×5 magnification)")
plt.xlabel("x  [m]")
plt.ylabel("y  [m]")
plt.axis("equal")
plt.tight_layout()
plt.savefig("q13_def.png", dpi=300)
plt.close()

# --------------------------------------------------------------------
# Plot von Mises stress on the deformed mesh
plt.figure(figsize=(8, 4))
p = plot(vm_proj, mesh=mesh_def, cmap="viridis")
plt.colorbar(p, label="von Mises stress [Pa]")
plt.title("von Mises stress on deformed configuration")
plt.xlabel("x  [m]")
plt.ylabel("y  [m]")
plt.axis("equal")
plt.tight_layout()
plt.savefig("q13_vm.png", dpi=300)
plt.close()