# filename: q11_boussinesq.py
from fenics import *
import numpy as np
import matplotlib.pyplot as plt

# -------------------- Parameters --------------------
rho   = 1.0               # kg/m^3
mu    = 1.5e-5            # Pa·s
nu    = mu / rho          # kinematic viscosity
alpha = 2.1e-5            # thermal diffusivity
gbeta = 3.15e-5           # g*beta (m·s⁻²·K⁻¹)
T_ref = 0.5               # K

# -------------------- Mesh & Function Spaces --------------------
mesh = UnitSquareMesh(64, 64)

V_el = VectorElement("P", mesh.ufl_cell(), 2)   # velocity (P2)
Q_el = FiniteElement("P", mesh.ufl_cell(), 1)   # pressure (P1)
T_el = FiniteElement("P", mesh.ufl_cell(), 1)   # temperature (P1)

W = FunctionSpace(mesh, MixedElement([V_el, Q_el, T_el]))

# -------------------- Boundary Definitions --------------------
class LeftWall(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and on_boundary

class RightWall(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 1.0) and on_boundary

class NoSlipWall(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

left  = LeftWall()
right = RightWall()
noslip = NoSlipWall()

# Mark boundaries (needed for Dirichlet BCs and for ds measure)
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
left.mark(boundaries, 1)
right.mark(boundaries, 2)
noslip.mark(boundaries, 3)

ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# -------------------- Trial / Test Functions --------------------
w = Function(W)               # (u, p, T) unknown
(u, p, T) = split(w)

(v, q, S) = TestFunctions(W)

# -------------------- Boundary Conditions --------------------
zero_vec = Constant((0.0, 0.0))
bc_u_left   = DirichletBC(W.sub(0), zero_vec, boundaries, 1)
bc_u_right  = DirichletBC(W.sub(0), zero_vec, boundaries, 2)
bc_u_topbot = DirichletBC(W.sub(0), zero_vec, boundaries, 3)

bc_T_left   = DirichletBC(W.sub(2), Constant(1.0), boundaries, 1)  # hot wall
bc_T_right  = DirichletBC(W.sub(2), Constant(0.0), boundaries, 2)  # cold wall

bcs = [bc_u_left, bc_u_right, bc_u_topbot, bc_T_left, bc_T_right]

# -------------------- Variational Forms --------------------
# Body force (Boussinesq buoyancy)
f = as_vector([Constant(0.0), gbeta*(T - T_ref)])

# Momentum equation (steady Navier–Stokes)
F_mom = (nu*inner(grad(u), grad(v))*dx
         + inner(dot(u, nabla_grad(u)), v)*dx
         - div(v)*p*dx
         + q*div(u)*dx
         - inner(f, v)*dx)

# Energy equation (steady advection–diffusion)
F_energy = (alpha*dot(grad(T), grad(S))*dx
            + dot(u, grad(T))*S*dx)

F = F_mom + F_energy

# -------------------- Non‑linear Solver (Newton) --------------------
solve(F == 0, w, bcs,
      solver_parameters={'newton_solver':
                         {'relative_tolerance': 1e-6,
                          'absolute_tolerance': 1e-8,
                          'maximum_iterations': 25,
                          'linear_solver': 'mumps'}})

# -------------------- Extract Solutions --------------------
(u_sol, p_sol, T_sol) = w.split()

# -------------------- Save to XDMF --------------------
xdmf = XDMFFile(mesh.mpi_comm(), "q11_solution.xdmf")
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True
xdmf.write(u_sol, 0.0)
xdmf.write(p_sol, 0.0)
xdmf.write(T_sol, 0.0)
xdmf.close()

# -------------------- Temperature Colormap --------------------
nx, ny = 200, 200
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)
T_vals = np.array([T_sol(Point(xi, yi)) for xi, yi in zip(np.ravel(X), np.ravel(Y))])
T_vals = T_vals.reshape((ny, nx))

plt.figure(figsize=(6,5))
c = plt.contourf(X, Y, T_vals, 50, cmap='inferno')
plt.colorbar(c, label='Temperature (K)')
plt.title('Steady Temperature Field')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.savefig('q11_T.png', dpi=300)
plt.close()

# -------------------- Average Nusselt Number at Left Wall --------------------
# Nusselt number Nu = -∂T/∂x at x = 0 (ΔT = 1, L = 1)
n = FacetNormal(mesh)
# For the left wall the outward normal points in the -x direction,
# so -dot(grad(T), n) = - (∂T/∂x * (-1)) = ∂T/∂x.
flux = -dot(grad(T_sol), n)          # heat flux (positive into the fluid)
Nu_avg = assemble(flux*ds(1))        # integrate over left boundary (marker 1)
print(f"Average Nusselt number at left wall: {Nu_avg:.6f}")