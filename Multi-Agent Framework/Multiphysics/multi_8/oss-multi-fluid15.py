# -*- coding: utf-8 -*-
#
# Coupled Stokes/Darcy flow in a two-region channel (legacy FEniCS)
#
#   Ω_f : Stokes (μ = 0.02 Pa·s)
#   Ω_p : Darcy  (K = 1e-6 m², κ = K/μ)
#
#   Interface Γ at x = 0.6 enforces:
#       u_f·n = - (K/μ) ∂_n p_f   (normal flux continuity)
#       p_f   = p_p               (pressure continuity)
#       u_f·t = 0                  (no-slip on fluid side)
#
#   Boundary conditions:
#       inlet  (x=0)      : u_f = (6 Ū y (H-y)/H² , 0)
#       fluid walls (y=0,H): u_f = (0,0)
#       porous walls (y=0,H): no-flux (∂_n p = 0)
#       outlet (x=1)      : p = 0
#
#   Output:
#       interface velocity profiles (normal & tangential) → q15_interface.csv
#       pressure field (both subdomains)               → q15_p.png
#       full solution (velocity + pressure)            → q15_solution.xdmf
#
# ---------------------------------------------------------------

from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1. Parameters & mesh
# ---------------------------------------------------------------
mu    = 0.02
K     = 1.0e-6
U_bar = 0.1
H     = 0.2
x_int = 0.6

nx, ny = 120, 40
mesh = RectangleMesh(Point(0.0, 0.0), Point(1.0, H), nx, ny, "crossed")

# ---------------------------------------------------------------
# 2. Sub-domain & facet markers (single cell marker)
# ---------------------------------------------------------------
cell_marker = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
for cell in cells(mesh):
    if cell.midpoint().x() < x_int - DOLFIN_EPS:
        cell_marker[cell] = 1
    else:
        cell_marker[cell] = 2

facet_marker = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
tol = 1e-10
for facet in facets(mesh):
    x = facet.midpoint().x()
    y = facet.midpoint().y()
    if near(x, 0.0, tol):
        facet_marker[facet] = 1
    elif near(x, 1.0, tol):
        facet_marker[facet] = 2
    elif near(y, 0.0, tol) or near(y, H, tol):
        facet_marker[facet] = 3

class Interface(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], x_int)

Interface().mark(facet_marker, 4)

# ---------------------------------------------------------------
# 3. Measures for sub-domains and interface
# ---------------------------------------------------------------
dx   = Measure('dx', domain=mesh, subdomain_data=cell_marker)
dS_i = Measure('dS', domain=mesh, subdomain_data=facet_marker)

# ---------------------------------------------------------------
# 4. Function spaces (Taylor–Hood for Stokes, P1 for pressure)
# ---------------------------------------------------------------
Ve = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Pe = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = MixedElement([Ve, Pe])
W  = FunctionSpace(mesh, TH)

# ---------------------------------------------------------------
# 5. Trial / test functions
# ---------------------------------------------------------------
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

# ---------------------------------------------------------------
# 6. Normal & tangential vectors on the interface
# ---------------------------------------------------------------
n_plus = FacetNormal(mesh)('+')
t = as_vector([0.0, 1.0])

# ---------------------------------------------------------------
# 7. Nitsche parameters
# ---------------------------------------------------------------
gamma_n = 10.0
gamma_t = 10.0
gamma_p = 10.0
h = CellDiameter(mesh)
h_avg = avg(h)

# ---------------------------------------------------------------
# 8. Weak forms (with side-restricted fields)
# ---------------------------------------------------------------
a_fluid = mu*inner(grad(u), grad(v))*dx(1) \
          - div(v)*p*dx(1) \
          - q*div(u)*dx(1)

kappa = K/mu
a_darcy = kappa*inner(grad(p), grad(q))*dx(2)
gamma_u = 1e6
a_darcy = a_darcy + gamma_u*inner(u, v)*dx(2)

u_f = u('+')
p_f = p('+')
u_p = u('-')
p_p = p('-')
v_f = v('+')
q_f = q('+')
q_m = q('-')
grad_u_f = grad(u)('+')
grad_v_f = grad(v)('+')
grad_q_f = grad(q)('+')
grad_p_m = grad(p)('-')

a_interface = (
    - mu*dot(grad_v_f*n_plus, n_plus)*(dot(u_f, n_plus) + kappa*dot(grad_p_m, n_plus))*dS_i(4)
    - mu*dot(grad_u_f*n_plus, n_plus)*(dot(v_f, n_plus) + kappa*dot(grad_q_f, n_plus))*dS_i(4)
    + gamma_n*mu/h_avg*(dot(u_f, n_plus) + kappa*dot(grad_p_m, n_plus))*
                 (dot(v_f, n_plus) + kappa*dot(grad_q_f, n_plus))*dS_i(4)
    + gamma_t*mu/h_avg*dot(u_f, t)*dot(v_f, t)*dS_i(4)
    - (p_f - p_p)*dot(v_f, n_plus)*dS_i(4)
    - (q_f - q_m)*(dot(u_f, n_plus) + kappa*dot(grad_p_m, n_plus))*dS_i(4)
    + gamma_p/h_avg*(p_f - p_p)*(q_f - q_m)*dS_i(4)
)

a = a_fluid + a_darcy + a_interface

L = Constant(0.0)*q*dx(1) + Constant(0.0)*q*dx(2)

# ---------------------------------------------------------------
# 9. Boundary conditions
# ---------------------------------------------------------------
u_in = Expression(("6.0*U_bar*x[1]*(H - x[1])/(H*H)", "0.0"),
                  U_bar=U_bar, H=H, degree=2)

bcs = [
    DirichletBC(W.sub(0), u_in, facet_marker, 1),
    DirichletBC(W.sub(0), Constant((0.0, 0.0)), facet_marker, 3),
    DirichletBC(W.sub(1), Constant(0.0), facet_marker, 2)
]

# ---------------------------------------------------------------
# 10. Solve the saddle-point system
# ---------------------------------------------------------------
w = Function(W)
solve(a == L, w, bcs, solver_parameters={"linear_solver": "mumps"})

(u_f, p) = w.split(deepcopy=True)

u_p = project(-kappa*grad(p), VectorFunctionSpace(mesh, "P", 1))

# ---------------------------------------------------------------
# 11. Interface velocity profiles (Γ at x = x_int)
# ---------------------------------------------------------------
y_vals = np.linspace(0.0, H, 200)
u_n_vals = np.zeros_like(y_vals)
u_t_vals = np.zeros_like(y_vals)

for i, y in enumerate(y_vals):
    pt = Point(x_int, y)
    u_n_vals[i] = u_f(pt)[0]
    u_t_vals[i] = u_f(pt)[1]

np.savetxt("q15_interface.csv",
           np.column_stack([y_vals, u_n_vals, u_t_vals]),
           header="y, u_normal, u_tangential", delimiter=",", comments="")

# ---------------------------------------------------------------
# 12. Pressure field plot (both subdomains)
# ---------------------------------------------------------------
plt.figure(figsize=(6, 4))
p_plot = plot(p, title="Pressure (Pa)", cmap="viridis")
plt.colorbar(p_plot)
plt.xlabel("x (m)"); plt.ylabel("y (m)")
plt.tight_layout()
plt.savefig("q15_p.png", dpi=300)
plt.close()

# ---------------------------------------------------------------
# 13. Save full solution (velocity + pressure) to XDMF
# ---------------------------------------------------------------
xdmf = XDMFFile(mesh.mpi_comm(), "q15_solution.xdmf")
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True
xdmf.write(u_f, 0.0)
xdmf.write(p, 0.0)
xdmf.close()