from dolfin import *
from math import pi

# ----------------------------------------------------------------------
# 1. Mesh (interface y = 0 is a mesh line)
# ----------------------------------------------------------------------
nx, ny = 32, 32
mesh = RectangleMesh(Point(0.0, -1.0), Point(pi, 1.0), nx, ny, "crossed")

# ----------------------------------------------------------------------
# 2. Cell markers (0: Stokes, 1: Darcy)
# ----------------------------------------------------------------------
domains = MeshFunction("size_t", mesh, mesh.topology().dim())
domains.set_all(0)
class DarcyDomain(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] < 0.0 + DOLFIN_EPS
DarcyDomain().mark(domains, 1)
dx = Measure("dx", domain=mesh, subdomain_data=domains)

# ----------------------------------------------------------------------
# 3. Facet markers (interface and external boundaries)
# ----------------------------------------------------------------------
facet_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)

class Interface(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0)
Interface().mark(facet_markers, 1)

class StokesBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[1] > DOLFIN_EPS
StokesBoundary().mark(facet_markers, 2)

class DarcyBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[1] < -DOLFIN_EPS
DarcyBoundary().mark(facet_markers, 3)

ds = Measure("ds", domain=mesh, subdomain_data=facet_markers)

# ----------------------------------------------------------------------
# 4. Mixed function space (Taylor–Hood + RT0)
# ----------------------------------------------------------------------
V_s = VectorFunctionSpace(mesh, "Lagrange", 2)   # Stokes velocity (P2)
Q_s = FunctionSpace(mesh, "Lagrange", 1)         # Stokes pressure (P1)
V_d = FunctionSpace(mesh, "RT", 1)               # Darcy velocity (H(div))
Q_d = FunctionSpace(mesh, "DG", 0)               # Darcy pressure (P0)

mixed_el = MixedElement([V_s.ufl_element(),
                         Q_s.ufl_element(),
                         V_d.ufl_element(),
                         Q_d.ufl_element()])
W = FunctionSpace(mesh, mixed_el)

# ----------------------------------------------------------------------
# 5. Trial / test functions
# ----------------------------------------------------------------------
(u_s, p_s, u_d, p_d) = TrialFunctions(W)
(v_s, q_s, v_d, q_d) = TestFunctions(W)

# ----------------------------------------------------------------------
# 6. Physical parameters
# ----------------------------------------------------------------------
g = rho = nu = mu = k = K = alpha = 1.0
beta = 10.0   # Nitsche penalty

# ----------------------------------------------------------------------
# 7. Exact Dirichlet data
# ----------------------------------------------------------------------
w_expr  = Expression("-1.0 - 0.5*x[1] + 0.25*x[1]*x[1]", degree=2)
dw_expr = Expression("-0.5 + 0.5*x[1]", degree=2)

u_s_exact = Expression(("dw*cos(x[0])", "w*sin(x[0])"),
                       w=w_expr, dw=dw_expr, degree=3)

p_d_exact = Expression("exp(x[1])*sin(x[0])", degree=3)

# ----------------------------------------------------------------------
# 8. Body force in Stokes region
# ----------------------------------------------------------------------
b = Expression((
    "0.5*x[1]*cos(x[0]) - 0.5*cos(x[0])",
    "(0.25*x[1]*x[1] - 0.5*x[1] - 1.5)*sin(x[0])"),
    degree=3)

# ----------------------------------------------------------------------
# 9. Dirichlet boundary conditions (facet markers)
# ----------------------------------------------------------------------
bcs = []
bcs.append(DirichletBC(W.sub(0), u_s_exact, facet_markers, 2))  # Stokes velocity
bcs.append(DirichletBC(W.sub(3), p_d_exact, facet_markers, 3))  # Darcy pressure

# ----------------------------------------------------------------------
# 10. Normal and tangential vectors on the interface
# ----------------------------------------------------------------------
n = FacetNormal(mesh)
t = as_vector((n[1], -n[0]))
h = CellDiameter(mesh)

# ----------------------------------------------------------------------
# 11. Variational forms
# ----------------------------------------------------------------------
# Stokes part
a_s_form = 2.0*nu*inner(sym(grad(u_s)), sym(grad(v_s)))
a_s_form -= p_s*div(v_s) + q_s*div(u_s)          # integrand
a_s = a_s_form*dx(0)
L_s = dot(b, v_s)*dx(0)

# Darcy part (RT0–P0)
a_d = (mu/k)*dot(u_d, v_d)*dx(1) - p_d*div(v_d)*dx(1) - q_d*div(u_d)*dx(1)

# Interface coupling (Nitsche + Beavers–Joseph)
sigma_s = -p_s*Identity(2) + 2.0*nu*sym(grad(u_s))

a_int = (-dot(sigma_s*n, v_s) + dot(p_d*n, v_s))*ds(1)
a_int += (beta/h)*dot(u_s - u_d, n)*dot(v_s, n)*ds(1)
a_int += (alpha/sqrt(k))*dot(u_s - u_d, t)*dot(v_s, t)*ds(1)

# Assemble total system
a = a_s + a_d + a_int
L = L_s

# ----------------------------------------------------------------------
# 12. Solve the linear system
# ----------------------------------------------------------------------
w_sol = Function(W)
solve(a == L, w_sol, bcs)

# ----------------------------------------------------------------------
# 13. Split solution
# ----------------------------------------------------------------------
(u_s_h, p_s_h, u_d_h, p_d_h) = w_sol.split(deepcopy=True)

# ----------------------------------------------------------------------
# 14. Write results (Stokes velocity & Darcy pressure)
# ----------------------------------------------------------------------
with XDMFFile(mesh.mpi_comm(), "stokes_darcy.xdmf") as xdmf:
    xdmf.write(u_s_h, 0.0)
    xdmf.write(p_d_h, 0.0)