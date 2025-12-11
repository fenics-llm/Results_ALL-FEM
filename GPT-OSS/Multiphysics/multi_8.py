# filename: q15_coupled_stokes_darcy.py
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import csv

# -------------------------------------------------
#  Parameters
# -------------------------------------------------
mu      = 0.02          # dynamic viscosity [Pa·s]
K       = 1.0e-6        # permeability [m²]
U_bar   = 0.1           # mean inlet speed [m/s]
H       = 0.2           # channel height [m]
L_f     = 0.6           # length of fluid sub‑domain
L_tot   = 1.0           # total length

# -------------------------------------------------
#  Mesh and sub‑domains
# -------------------------------------------------
Nx, Ny = 120, 40
mesh = RectangleMesh(Point(0.0, 0.0), Point(L_tot, H), Nx, Ny)

class Fluid(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] <= L_f + DOLFIN_EPS
class Porous(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] >= L_f - DOLFIN_EPS
class Interface(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], L_f)

subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
subdomains.set_all(0)
Fluid().mark(subdomains, 1)      # fluid = 1
Porous().mark(subdomains, 2)     # porous = 2

dx = Measure("dx", domain=mesh, subdomain_data=subdomains)

facets = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
facets.set_all(0)
Interface().mark(facets, 1)     # interface = 1
ds = Measure("ds", domain=mesh, subdomain_data=facets)

n = FacetNormal(mesh)           # outward normal (fluid → porous)

# -------------------------------------------------
#  1) Stokes problem in the fluid region Ω_f
# -------------------------------------------------
Vf = VectorFunctionSpace(mesh, "CG", 2)
Pf = FunctionSpace(mesh, "CG", 1)
Wf = FunctionSpace(mesh, MixedElement([Vf.ufl_element(),
                                      Pf.ufl_element()]))

(u_f, p_f) = TrialFunctions(Wf)
(v_f, q_f) = TestFunctions(Wf)

# weak Stokes (integrated only over subdomain 1)
a_stokes = (mu*inner(grad(u_f), grad(v_f))
            - div(v_f)*p_f
            - q_f*div(u_f))*dx(1)
L_stokes = Constant(0.0)*v_f[0]*dx(1)   # zero RHS

# inlet parabolic velocity (x = 0)
inlet = Expression(("6.0*U_bar*x[1]*(H - x[1])/pow(H,2)", "0.0"),
                   degree=2, U_bar=U_bar, H=H)

def inlet_bc(x, on_boundary):
    return on_boundary and near(x[0], 0.0)
bc_inlet = DirichletBC(Wf.sub(0), inlet, inlet_bc)

# no‑slip walls (y = 0, y = H)
def walls_bc(x, on_boundary):
    return on_boundary and (near(x[1], 0.0) or near(x[1], H))
bc_walls = DirichletBC(Wf.sub(0), Constant((0.0, 0.0)), walls_bc)

# pressure pinning to remove the null‑space (pointwise at (0,0))
def pressure_pin(x, on_boundary):
    return near(x[0], 0.0) and near(x[1], 0.0)
bc_pin = DirichletBC(Wf.sub(1), Constant(0.0), pressure_pin,
                    method="pointwise")

bcs_stokes = [bc_inlet, bc_walls, bc_pin]

w_f = Function(Wf)
solve(a_stokes == L_stokes, w_f, bcs_stokes,
      solver_parameters={"linear_solver": "mumps"})
(u_f_sol, p_f_sol) = w_f.split()

# -------------------------------------------------
#  2) Darcy problem in the porous region Ω_p
# -------------------------------------------------
Pp = FunctionSpace(mesh, "CG", 1)
p_p = TrialFunction(Pp)
q_p = TestFunction(Pp)

# weak Darcy (integrated only over subdomain 2)
a_darcy = (K/mu)*inner(grad(p_p), grad(q_p))*dx(2)

# Neumann condition on the interface: (K/μ)∂p/∂n = -u_f·n
# → RHS = ∫_Γ u_f·n q ds
g_interface = dot(u_f_sol, n)          # scalar on the interface
L_darcy = g_interface*q_p*ds(1)

# outlet pressure (x = 1) → Dirichlet p = 0
def outlet_bc(x, on_boundary):
    return on_boundary and near(x[0], L_tot)
bc_outlet = DirichletBC(Pp, Constant(0.0), outlet_bc)

p_p_sol = Function(Pp)
solve(a_darcy == L_darcy, p_p_sol, [bc_outlet],
      solver_parameters={"linear_solver": "mumps"})

# Darcy velocity (only needed for output)
u_p_sol = project(-(K/mu)*grad(p_p_sol), Vf)

# -------------------------------------------------
#  3) Post‑processing
# -------------------------------------------------
# ---- Interface velocity profile (fluid side) ----
y_vals = np.linspace(0.0, H, 200)
ux_vals = []   # normal component (x‑direction)
uy_vals = []   # tangential component (y‑direction)

for y in y_vals:
    pt = Point(L_f, y)
    ux, uy = u_f_sol(pt)
    ux_vals.append(ux)
    uy_vals.append(uy)

with open("q15_interface.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["y", "u_normal (ux)", "u_tangential (uy)"])
    for y, ux, uy in zip(y_vals, ux_vals, uy_vals):
        writer.writerow([y, ux, uy])

# ---- Combined pressure field (fluid pressure in Ω_f, porous pressure in Ω_p) ----
P_comb = FunctionSpace(mesh, "CG", 1)
p_comb = Function(P_comb)
p_comb.vector()[:] = p_f_sol.vector() + p_p_sol.vector()   # each is non‑zero only in its sub‑domain

# Plot pressure
plt.figure(figsize=(6, 4))
c = plot(p_comb, title="Pressure field", cmap="viridis")
plt.colorbar(c)
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.tight_layout()
plt.savefig("q15_p.png", dpi=300)
plt.close()

# ---- Build a single velocity field for the whole domain ----
V_all = VectorFunctionSpace(mesh, "CG", 2)
u_all = Function(V_all)

class CombinedVelocity(UserExpression):
    def __init__(self, u_f, u_p, subdomains, **kwargs):
        super().__init__(**kwargs)
        self.u_f = u_f
        self.u_p = u_p
        self.subdomains = subdomains
        self.tree = mesh.bounding_box_tree()
    def eval(self, values, x):
        cell_id = self.tree.compute_first_entity_collision(Point(x))
        sub = self.subdomains[cell_id]
        if sub == 1:                     # fluid
            values[:] = self.u_f(x)
        else:                            # porous
            values[:] = self.u_p(x)
    def value_shape(self):
        return (2,)

u_comb_expr = CombinedVelocity(u_f_sol, u_p_sol, subdomains, degree=2)
u_all.interpolate(u_comb_expr)

# ---- Save results (XDMF) ----
xdmf = XDMFFile(mesh.mpi_comm(), "q15_solution.xdmf")
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True
xdmf.write(u_all, 0.0)
xdmf.write(p_comb, 0.0)
xdmf.close()

print("Computation finished. Files generated:")
print("- q15_interface.csv")
print("- q15_p.png")
print("- q15_solution.xdmf")