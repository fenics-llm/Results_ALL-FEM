# filename: q15_coupled_stokes_darcy.py
from dolfin import *
import numpy as np

# ------------------- Parameters -------------------
mu    = 0.02          # fluid viscosity
K     = 1.0e-6        # permeability
U_bar = 0.1           # mean inlet velocity
H     = 0.2           # channel height
x_int = 0.6           # interface location

# ------------------- Mesh & Subdomains -------------------
nx, ny = 120, 40
mesh = RectangleMesh(Point(0.0, 0.0), Point(1.0, H), nx, ny)

# Cell markers: 1 = fluid, 2 = porous
subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
subdomains.set_all(0)
class FluidDomain(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] <= x_int + DOLFIN_EPS
class PorousDomain(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] >= x_int - DOLFIN_EPS
FluidDomain().mark(subdomains, 1)
PorousDomain().mark(subdomains, 2)

dx = Measure('dx', domain=mesh, subdomain_data=subdomains)

# Facet markers for boundaries and interface
facet_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
facet_markers.set_all(0)
class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and on_boundary
class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 1.0) and on_boundary
class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[1], 0.0) or near(x[1], H)) and on_boundary
class Interface(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], x_int) and on_boundary
Inlet().mark(facet_markers, 1)
Outlet().mark(facet_markers, 2)
Walls().mark(facet_markers, 3)
Interface().mark(facet_markers, 4)

ds = Measure('ds', domain=mesh, subdomain_data=facet_markers)

# ------------------- Function Spaces -------------------
V_f = VectorElement("Lagrange", mesh.ufl_cell(), 2)   # Stokes velocity
Q_f = FiniteElement("Lagrange", mesh.ufl_cell(), 1)   # Stokes pressure
Q_p = FiniteElement("Lagrange", mesh.ufl_cell(), 1)   # Darcy pressure
TH  = MixedElement([V_f, Q_f, Q_p])
W   = FunctionSpace(mesh, TH)

# ------------------- Trial / Test Functions -------------------
(u_f, p_f, p_p) = TrialFunctions(W)
(v_f, q_f, q_p) = TestFunctions(W)

# Normal and tangential vectors on the interface
n = FacetNormal(mesh)
t = as_vector([n[1], -n[0]])   # rotate n by +90° in 2D

# ------------------- Inlet Velocity Profile -------------------
U_in = Expression(("6.0*U_bar*x[1]*(H - x[1])/(H*H)", "0.0"),
                  U_bar=U_bar, H=H, degree=2)

# ------------------- Boundary Conditions -------------------
bcs = []
bcs.append(DirichletBC(W.sub(0), U_in, facet_markers, 1))                     # inlet velocity
bcs.append(DirichletBC(W.sub(0), Constant((0.0, 0.0)), facet_markers, 3))    # fluid walls
bcs.append(DirichletBC(W.sub(0), Constant((0.0, 0.0)), facet_markers, 4))    # u_f·t = 0 on interface
bcs.append(DirichletBC(W.sub(2), Constant(0.0), facet_markers, 2))           # outlet pressure

# ------------------- Weak Forms -------------------
# Stokes (fluid) part
a_f = ((2.0*mu*inner(sym(grad(u_f)), sym(grad(v_f)))
        - div(v_f)*p_f - q_f*div(u_f))*dx(1)

# Darcy (porous) part (pressure formulation)
a_p = (K/mu)*inner(grad(p_p), grad(q_p))*dx(2)

# Interface coupling (penalty/Nitsche)
beta  = 1e2*mu/K          # normal flux penalty
gamma = 1e2*mu/K          # pressure continuity penalty
eta   = 1e2*mu/K          # tangential velocity penalty

flux_f   = dot(u_f, n) + (K/mu)*dot(grad(p_p), n)          # u_f·n + (K/μ)∂p_p/∂n
flux_test = dot(v_f, n) + (K/mu)*dot(grad(q_p), n)

a_int = (beta*flux_f*flux_test
         + gamma*(p_f - p_p)*(q_f - q_p)
         + eta*dot(u_f, t)*dot(v_f, t))*ds(4)

# Total bilinear form
a = a_f + a_p + a_int

# RHS (zero body forces)
L = inner(Constant((0.0, 0.0)), v_f)*dx(1) + Constant(0.0)*q_f*dx(1) + Constant(0.0)*q_p*dx(2)

# ------------------- Solve -------------------
w = Function(W)
solve(a == L, w, bcs)

# ------------------- Extract Solutions -------------------
(u_f_sol, p_f_sol, p_p_sol) = w.split(deepcopy=True)

# Darcy velocity from pressure
V_vec = VectorFunctionSpace(mesh, "CG", 1)
u_p_sol = project(-(K/mu)*grad(p_p_sol), V_vec)

# ------------------- Save to XDMF -------------------
xdmf = XDMFFile(mesh.mpi_comm(), "q15_solution.xdmf")
xdmf.write(u_f_sol, 0.0)
xdmf.write(u_p_sol, 0.0)
xdmf.write(p_f_sol, 0.0)
xdmf.write(p_p_sol, 0.0)
xdmf.close()

# ------------------- Interface Profiles -------------------
y_vals = np.linspace(0.0, H, 200)
ux_vals = np.zeros_like(y_vals)
uy_vals = np.zeros_like(y_vals)
for i, y in enumerate(y_vals):
    point = (x_int, y)
    ux_vals[i] = u_f_sol(point)[0]   # normal (x) component
    uy_vals[i] = u_f_sol(point)[1]   # tangential (y) component
np.savetxt("q15_interface.csv",
           np.column_stack((y_vals, ux_vals, uy_vals)),
           delimiter=",",
           header="y, u_n, u_t",
           comments="")

# ------------------- Pressure Colour Map -------------------
Vcg = FunctionSpace(mesh, "CG", 1)
p_comb = Function(Vcg)
cell_domains = subdomains.array()
p_comb.vector()[cell_domains == 1] = p_f_sol.vector()[cell_domains == 1]
p_comb.vector()[cell_domains == 2] = p_p_sol.vector()[cell_domains == 2]
File("q15_p.pvd") << p_comb

# ------------------- Plot Pressure (headless) -------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.figure()
c = plot(p_comb, title="Pressure")
plt.colorbar(c)
plt.savefig("q15_p.png")
plt.close()