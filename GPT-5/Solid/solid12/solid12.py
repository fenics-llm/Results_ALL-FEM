from __future__ import print_function
from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# Geometry and mesh (units: metres)
# -----------------------------------------------------------------------------
Lx, Ly = 1.0, 0.20
cx, cy, a = 0.50, 0.10, 0.04

domain = Rectangle(Point(0.0, 0.0), Point(Lx, Ly)) - Circle(Point(cx, cy), a)
# A reasonably fine mesh; adjust for accuracy vs. cost
mesh = generate_mesh(domain, 120)

# -----------------------------------------------------------------------------
# Function spaces (Taylor–Hood: P2 for u, P1 for p)
# -----------------------------------------------------------------------------
Ve = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Pe = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, MixedElement([Ve, Pe]))

# Trial/unknown and test functions
w = Function(W)
(u, p) = split(w)
(v, q) = TestFunctions(W)

# -----------------------------------------------------------------------------
# Material parameters (incompressible neo-Hookean, plane strain setting)
# -----------------------------------------------------------------------------
E  = 5.0e6          # Pa
nu = 0.5
mu = E / (2.0*(1.0 + nu))  # Shear modulus (Pa)

I = Identity(2)
F = I + grad(u)
C = F.T*F
Ic = tr(C)
J = det(F)
invFT = inv(F).T

# Isochoric neo-Hookean energy (2D trace uses "- 2" instead of "- 3")
psi_iso = (mu/2.0)*(J**(-2.0/3.0)*Ic - 2.0)

# Mixed (u, p) formulation: Lagrange multiplier enforces J = 1
Pi = psi_iso*dx - p*(J - 1.0)*dx   # Total potential (no external traction work)

# First variation (residual) and Jacobian for Newton
Res = derivative(Pi, w, TestFunction(W))
Jac = derivative(Res, w, TrialFunction(W))

# -----------------------------------------------------------------------------
# Boundary conditions
# -----------------------------------------------------------------------------
tol = 1e-14

class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, tol)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], Lx, tol)

left  = Left()
right = Right()

u_fix = Constant((0.0, 0.0))
u_pull = Constant((0.060, 0.0))  # prescribed displacement at x = 1.0

bc_left  = DirichletBC(W.sub(0), u_fix, left)
bc_right = DirichletBC(W.sub(0), u_pull, right)
bcs = [bc_left, bc_right]  # pressure is free (incompressible, traction-free elsewhere)

# -----------------------------------------------------------------------------
# Solve nonlinear problem
# -----------------------------------------------------------------------------
problem = NonlinearVariationalProblem(Res, w, bcs, Jac)
solver  = NonlinearVariationalSolver(problem)

prm = solver.parameters
prm["newton_solver"]["absolute_tolerance"] = 1e-10
prm["newton_solver"]["relative_tolerance"] = 1e-8
prm["newton_solver"]["maximum_iterations"] = 50
prm["newton_solver"]["linear_solver"] = "mumps"
prm["newton_solver"]["report"] = True

solver.solve()

(u_sol, p_sol) = w.split(True)  # deep copy for post-processing

# -----------------------------------------------------------------------------
# Post-processing: Cauchy stress and von Mises stress
# -----------------------------------------------------------------------------
F = I + grad(u_sol)
J = det(F)
Ic = tr(F.T*F)
invFT = inv(F).T

# First Piola–Kirchhoff stress for isochoric NH + pressure
P_tens = mu*J**(-2.0/3.0)*(F - (Ic/3.0)*invFT) + p_sol*J*invFT
sigma = (1.0/J) * P_tens * F.T  # Cauchy stress (2x2)

# 2D deviatoric part and von Mises (plane strain surrogate from 2D tensor)
tr_sigma = tr(sigma)
dev2 = sigma - (tr_sigma/2.0)*I
von_Mises = sqrt(3.0/2.0*inner(dev2, dev2))

# Project for smooth visualisation
P1  = FunctionSpace(mesh, "Lagrange", 1)
p_viz  = project(p_sol, P1)
vm_viz = project(von_Mises, P1)

# -----------------------------------------------------------------------------
# Save displacement (XDMF)
# -----------------------------------------------------------------------------
with XDMFFile(mesh.mpi_comm(), "q12_disp.xdmf") as xdmf:
    xdmf.parameters["functions_share_mesh"] = True
    xdmf.parameters["rewrite_function_mesh"] = True
    xdmf.write(u_sol)

# -----------------------------------------------------------------------------
# Static colour maps (PNG) for pressure and von Mises
# -----------------------------------------------------------------------------
# Extract vertex-wise data for contour plots
coords = mesh.coordinates()
triangles = np.asarray([cell.entities(0) for cell in cells(mesh)], dtype=np.int32)

def save_tricontour(fname, field, title, cb_label):
    vals = field.compute_vertex_values(mesh)
    fig = plt.figure(figsize=(7, 2.0*7*Ly/Lx))  # keep aspect reasonable
    ax = plt.gca()
    tcf = ax.tricontourf(coords[:,0], coords[:,1], triangles, vals, 100)
    ax.set_aspect('equal')
    ax.set_xlim(0.0, Lx); ax.set_ylim(0.0, Ly)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title(title)
    cbar = plt.colorbar(tcf, ax=ax)
    cbar.set_label(cb_label)
    # draw the hole outline for clarity
    theta = np.linspace(0, 2*np.pi, 361)
    ax.plot(cx + a*np.cos(theta), cy + a*np.sin(theta), 'k-', linewidth=0.75)
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close(fig)

save_tricontour("q12_p.png",  p_viz,  "Hydrostatic pressure p", "p (Pa)")
save_tricontour("q12_vm.png", vm_viz, "von Mises stress", "σ_vm (Pa)")

print("Done. Wrote q12_disp.xdmf, q12_p.png, q12_vm.png")