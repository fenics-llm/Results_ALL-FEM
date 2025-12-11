# -*- coding: utf-8 -*-
#
#  Coupled Navier–Stokes / Brinkman flow in a 2-D channel with a porous insert
#  (legacy FEniCS – dolfin)
#
import dolfin as dl
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dolfin import Point  # for pressure-drop probes

# ----------------------------------------------------------------------
# 1. Geometry and mesh
# ----------------------------------------------------------------------
L = 1.0          # channel length (m)
H = 0.20         # channel height (m)
nx, ny = 200, 40  # mesh resolution
mesh = dl.RectangleMesh(dl.Point(0.0, 0.0), dl.Point(L, H), nx, ny, "crossed")

# ----------------------------------------------------------------------
# 2. Subdomain markers (fluid = 0, porous = 1)
# ----------------------------------------------------------------------
subdomains = dl.MeshFunction("size_t", mesh, mesh.topology().dim())
subdomains.set_all(0)  # fluid by default

class Porous(dl.SubDomain):
    def inside(self, x, on_boundary):
        return dl.between(x[0], (0.40, 0.60)) and dl.between(x[1], (0.0, H))

Porous().mark(subdomains, 1)
dx = dl.Measure("dx", domain=mesh, subdomain_data=subdomains)

# ----------------------------------------------------------------------
# 3. Facet markers for external boundaries
# ----------------------------------------------------------------------
facets = dl.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
facets.set_all(0)

class Inlet(dl.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and dl.near(x[0], 0.0)

class Walls(dl.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (dl.near(x[1], 0.0) or dl.near(x[1], H))

class Outlet(dl.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and dl.near(x[0], L)

class PressurePoint(dl.SubDomain):
    def inside(self, x, on_boundary):
        return dl.near(x[0], 0.0) and dl.near(x[1], 0.0)

Inlet().mark(facets, 1)
Walls().mark(facets, 2)
Outlet().mark(facets, 3)
ds = dl.Measure("ds", domain=mesh, subdomain_data=facets)

# ----------------------------------------------------------------------
# 4. Physical parameters
# ----------------------------------------------------------------------
rho   = 1.0          # kg/m^3
mu    = 0.01         # Pa·s
K     = 1.0e-6       # m^2
alpha = mu / K        # Darcy drag coefficient
U_bar = 1.0          # inlet mean velocity (m/s)

# ----------------------------------------------------------------------
# 5. Mixed Taylor–Hood space (P2-P1)
# ----------------------------------------------------------------------
Ve = dl.VectorElement("Lagrange", mesh.ufl_cell(), 2)
Pe = dl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = dl.MixedElement([Ve, Pe])
W  = dl.FunctionSpace(mesh, TH)

# ----------------------------------------------------------------------
# 6. Boundary conditions
# ----------------------------------------------------------------------
# scalar ux profile (y = x[1])
ux_inlet = dl.Expression(
    "6.0*U_bar*x[1]*(H - x[1])/pow(H,2)",
    degree=2, U_bar=U_bar, H=H
)

bcs = [
    dl.DirichletBC(W.sub(0).sub(0), ux_inlet, facets, 1),          # ux on inlet
    dl.DirichletBC(W.sub(0).sub(1), dl.Constant(0.0), facets, 1), # uy on inlet
    dl.DirichletBC(W.sub(0).sub(0), dl.Constant(0.0), facets, 2), # ux on walls
    dl.DirichletBC(W.sub(0).sub(1), dl.Constant(0.0), facets, 2)  # uy on walls
    # outlet: natural (traction-free) – no Dirichlet needed
]
bc_p = dl.DirichletBC(W.sub(1), dl.Constant(0.0), PressurePoint(), 'pointwise')
bcs.append(bc_p)

# ----------------------------------------------------------------------
# 7. Variational formulation (Picard linearisation of convection)
# ----------------------------------------------------------------------
(u, p) = dl.TrialFunctions(W)
(v, q) = dl.TestFunctions(W)

def epsilon(v):
    return dl.sym(dl.grad(v))

# Viscous term (everywhere)
a_visc = 2.0 * mu * dl.inner(epsilon(u), epsilon(v)) * dx

# Darcy drag (only in porous region)
a_drag = alpha * dl.inner(u, v) * dx(1)

# Pressure–divergence coupling (both subdomains)
b_term = -p * dl.div(v) * dx
c_term = -q * dl.div(u) * dx

# Right-hand side – zero traction at outlet (scalar linear form)
L_form = dl.Constant(0.0) * dl.inner(v, dl.Constant((0.0, 0.0))) * ds(3)

# ----------------------------------------------------------------------
# 8. Picard iteration for the convective term
# ----------------------------------------------------------------------
w = dl.Function(W)                     # (u,p) solution
u_prev = dl.Function(W.sub(0).collapse())  # start from zero velocity

tol = 1e-6
max_iter = 20
for it in range(max_iter):
    # convection term linearised with u_prev: (u_prev·∇)u · v
    a_conv = rho * dl.inner(dl.dot(u_prev, dl.grad(u)), v) * dx(0)
    a = a_visc + a_drag + a_conv + b_term + c_term

    A = dl.assemble(a, keep_diagonal=True)
    b_vec = dl.assemble(L_form)
    for bc in bcs:
        bc.apply(A, b_vec)

    dl.solve(A, w.vector(), b_vec, "lu")

    u_sol, p_sol = w.split(deepcopy=True)

    # convergence check (velocity L2 norm)
    if it > 0:
        du = dl.Function(u_sol.function_space())
        du.vector()[:] = u_sol.vector() - u_prev.vector()
        rel = du.vector().norm('l2')/max(1.0, u_prev.vector().norm('l2'))
        if rel < 1e-8:
            break
    u_prev.assign(u_sol)   # update for next iteration

# ----------------------------------------------------------------------
# 9. Post-processing
# ----------------------------------------------------------------------
# Velocity magnitude
speed = dl.project(dl.sqrt(dl.inner(u_sol, u_sol)),
                  dl.FunctionSpace(mesh, "P", 2))

plt.figure(figsize=(8, 2))
c = dl.plot(speed, cmap='viridis')
plt.colorbar(c, label='|u| (m/s)')
plt.title('Velocity magnitude')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.tight_layout()
plt.savefig('q14_speed.png', dpi=300)
plt.close()

# Pressure drop across porous block (centreline y = H/2)
y_mid = H / 2.0
x_left  = 0.40 - 1e-6
x_right = 0.60 + 1e-6
p_left  = p_sol(Point(x_left,  y_mid))
p_right = p_sol(Point(x_right, y_mid))
dp = float(p_left - p_right)

with open('q14_dp.txt', 'w') as f:
    f.write(f'{dp:.6e}\n')

# Export full solution (velocity + pressure) to XDMF
with dl.XDMFFile(mesh.mpi_comm(), 'q14_solution.xdmf') as xdmf:
    xdmf.write(u_sol, 0.0)
    xdmf.write(p_sol, 0.0)

print('Simulation completed.')