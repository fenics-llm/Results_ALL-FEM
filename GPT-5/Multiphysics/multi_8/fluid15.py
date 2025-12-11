# q15_stokes_darcy_bjs_legacy.py
# FEniCS (legacy) monolithic Stokes–Brinkman with BJS slip at x=0.6
from dolfin import *
from mshr import *
import numpy as np

# -----------------------
# Parameters (SI units)
# -----------------------
mu   = Constant(0.02)          # dynamic viscosity [Pa·s]
K    = Constant(1.0e-6)        # permeability [m^2]
alpha_bjs = Constant(1.0)      # BJS slip coefficient [-]
Ubar = 0.1                     # mean inlet speed [m/s]
H    = 0.2                     # channel height [m]
L    = 1.0                     # channel length [m]
x_if = 0.6                     # interface x-position [m]

# -----------------------
# Mesh and subdomains
# -----------------------
# Unstructured triangular mesh via CSG with refinement near the interface
nx, ny = 160, 32  # base resolution (refine further if needed)

domain = Rectangle(Point(0.0, 0.0), Point(L, H))
# Build and refine near the interface strip to capture BJS and jump in coeffs
mesh = generate_mesh(domain, nx)

# Mark subdomains: 1 = fluid (x in [0,0.6]), 2 = porous (x in [0.6,1.0])
subdomains = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
class Fluid(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] <= x_if + DOLFIN_EPS
class Porous(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] >= x_if - DOLFIN_EPS

Fluid().mark(subdomains, 1)
Porous().mark(subdomains, 2)

# Mark boundaries
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0)
class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], L)
class WallBottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0)
class WallTop(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], H)
class Interface(SubDomain):
    def inside(self, x, on_boundary):
        # Interior interface is NOT on_boundary; we mark facets later via MeshFunction on facets
        return False

Inlet().mark(boundaries, 10)
Outlet().mark(boundaries, 20)
WallBottom().mark(boundaries, 30)
WallTop().mark(boundaries, 40)

# Mark the internal interface facets (x = x_if) using a facet function
facets = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
class InternalInterface(SubDomain):
    def inside(self, x, on_boundary):
        # mark facets whose midpoint is at x = x_if (within tolerance) AND not external boundary
        return near(x[0], x_if) and not on_boundary
InternalInterface().mark(facets, 99)

dxm = Measure("dx", domain=mesh, subdomain_data=subdomains)
ds  = Measure("ds", domain=mesh, subdomain_data=boundaries)
dS  = Measure("dS", domain=mesh, subdomain_data=facets)   # interior facets measure

# -----------------------
# Function spaces
# -----------------------
# Taylor–Hood: P2 velocity, P1 pressure
Ve = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Pe = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
W  = FunctionSpace(mesh, MixedElement([Ve, Pe]))

(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

# -----------------------
# Inflow profile (parabolic with mean Ubar)
# ux(y) = 6 * Ubar * y * (H - y) / H^2 ; uy = 0
# -----------------------
inlet_profile = Expression(("6.0*Ubar*x[1]*(H - x[1])/(H*H)", "0.0"),
                           Ubar=Ubar, H=H, degree=2)

# -----------------------
# Coefficients per subdomain
# Stokes in fluid (Ω_f): standard viscous term
# Brinkman in porous (Ω_p): viscous + drag (mu/K)*u
# -----------------------
I = Identity(mesh.geometry().dim())
def eps(w):
    return sym(grad(w))

# Drag coefficient (mu/K) active only in porous
drag = mu/K

chi_fluid  = conditional(eq(subdomains, 1), 1.0, 0.0)
chi_porous = conditional(eq(subdomains, 2), 1.0, 0.0)

# -----------------------
# Variational form (bulk)
# -----------------------
a_stokes = ( 2.0*mu*inner(eps(u), eps(v))*dxm(1)
           - div(v)*p*dxm(1)
           - q*div(u)*dxm(1) )

a_brink  = ( 2.0*mu*inner(eps(u), eps(v))*dxm(2)
           + drag*inner(u, v)*dxm(2)
           - div(v)*p*dxm(2)
           - q*div(u)*dxm(2) )

a = a_stokes + a_brink

L = Constant(0.0)*q*dx  # no body force; modify here if you add forcing

# -----------------------
# Interface: Beavers–Joseph–Saffman slip on fluid side
#   mu * (du_t/dn) = -(alpha * mu / sqrt(K)) * u_t
# Weakly: add Robin-like term on the interface for tangential component.
# We project tangential via (I - n⊗n).
# -----------------------
n = FacetNormal(mesh)
Nt = I - outer(n, n)  # tangential projector

bjs_coeff = alpha_bjs*mu/sqrt(K)
# Robin term: ∫_Γ bjs_coeff * (u_t · v_t) dS
a += bjs_coeff * inner(Nt*u, Nt*v) * dS(99)

# -----------------------
# Boundary conditions
# -----------------------
bcs = []

# Inlet velocity on fluid segment at x=0 (on both walls range, but only fluid exists there)
bc_in = DirichletBC(W.sub(0), inlet_profile, boundaries, 10)
bcs.append(bc_in)

# Fluid walls (y=0, y=H) on fluid side: no-slip
# We will enforce u=0 on full walls; this also applies to porous walls.
# For strictly "no-flux" in porous, u·n=0 would suffice, but Dirichlet u=0 is a safe impermeable choice.
bc_wb = DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 30)
bc_wt = DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 40)
bcs += [bc_wb, bc_wt]

# Outlet pressure p = 0 at x=L; classical do-nothing for u with fixed gauge pressure
bc_out_p = DirichletBC(W.sub(1), Constant(0.0), boundaries, 20)
bcs.append(bc_out_p)

# -----------------------
# Assemble & solve
# -----------------------
w = Function(W)
solve(a == L, w, bcs,
      solver_parameters={"linear_solver": "mumps"})

u_h, p_h = w.split(deepcopy=True)
u_h.rename("u", "velocity")
p_h.rename("p", "pressure")

# -----------------------
# Post-processing: interface profiles at x = 0.6
# normal n points from fluid to porous (here take n = (1,0) on the interface line)
# normal velocity = u_x ; tangential velocity = u_y for the vertical interface
# -----------------------
# Sample along the line x = x_if, y in [0,H]
n_samples = 401
ys = np.linspace(0.0, H, n_samples)
ux_vals = []
uy_vals = []
for yv in ys:
    pt = Point(x_if, yv)
    # Use 'closest cell' evaluation to be robust at interface
    ux_vals.append(u_h(pt)[0])
    uy_vals.append(u_h(pt)[1])

# Save to CSV: y, u_n(=ux), u_t(=uy)
import csv
with open("q15_interface.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["y", "u_normal(ux)", "u_tangential(uy)"])
    for yv, unv, utv in zip(ys, ux_vals, uy_vals):
        writer.writerow([float(yv), float(unv), float(utv)])

# -----------------------
# Save full solution to XDMF
# -----------------------
xdmf = XDMFFile(mesh.mpi_comm(), "q15_solution.xdmf")
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True
xdmf.write(u_h, 0.0)
xdmf.write(p_h, 0.0)
xdmf.close()

# -----------------------
# Pressure colour map as PNG
# -----------------------
# We will sample pressure on a uniform grid and plot via matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

nxp, nyp = 400, 80
xs = np.linspace(0.0, L, nxp)
ys = np.linspace(0.0, H, nyp)
P = np.zeros((nyp, nxp))
for j, yv in enumerate(ys):
    for i, xv in enumerate(xs):
        P[j, i] = p_h(Point(xv, yv))

plt.figure(figsize=(7, 1.6))
extent = [0.0, L, 0.0, H]
plt.imshow(P, origin="lower", extent=extent, aspect="auto")
plt.colorbar(label="pressure (Pa)")
plt.title("Pressure field p")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.tight_layout()
plt.savefig("q15_p.png", dpi=200)
plt.close()

# -----------------------
# Console prints
# -----------------------
print("Saved:")
print("  - q15_interface.csv  (y, u_normal=ux, u_tangential=uy along x=0.6)")
print("  - q15_solution.xdmf  (u, p)")
print("  - q15_p.png          (pressure colour map)")