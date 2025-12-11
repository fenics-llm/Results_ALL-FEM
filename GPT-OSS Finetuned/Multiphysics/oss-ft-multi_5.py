# filename: fsi_tube.py
from dolfin import *
from math import pi, cos

# -------------------------------------------------
# 1. Geometry & subdomain/facet markers
# -------------------------------------------------
L, H, t_wall = 6.0e-2, 1.0e-2, 0.1e-2          # cm
Nx, Ny_f, Ny_s = 60, 30, 6
Ny_total = Ny_f + 2*Ny_s
mesh = RectangleMesh(Point(0.0, -t_wall), Point(L, H + t_wall), Nx, Ny_total)

fluid_marker = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
solid_marker = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)

class FluidDomain(SubDomain):
    def inside(self, x, on_boundary):
        return (x[1] >= -DOLFIN_EPS) and (x[1] <= H + DOLFIN_EPS)

FluidDomain().mark(fluid_marker, 1)
solid_marker.array()[fluid_marker.array() == 0] = 2

facet_marker = MeshFunction('size_t', mesh, mesh.topology().dim() - 1, 0)

class Inlet(SubDomain):    def inside(self, x, on_boundary): return on_boundary and near(x[0], 0.0)
class Outlet(SubDomain):   def inside(self, x, on_boundary): return on_boundary and near(x[0], L)
class Bottom(SubDomain):   def inside(self, x, on_boundary): return on_boundary and near(x[1], 0.0)
class Top(SubDomain):      def inside(self, x, on_boundary): return on_boundary and near(x[1], H)

Inlet().mark(facet_marker, 1)
Outlet().mark(facet_marker, 2)
Bottom().mark(facet_marker, 3)
Top().mark(facet_marker, 4)

dx_f = Measure('dx', domain=mesh, subdomain_data=fluid_marker)
dx_s = Measure('dx', domain=mesh, subdomain_data=solid_marker)
ds   = Measure('ds', domain=mesh, subdomain_data=facet_marker)

# -------------------------------------------------
# 2. Function spaces
# -------------------------------------------------
V_f = VectorFunctionSpace(mesh, "Lagrange", 2)   # velocity
Q_f = FunctionSpace(mesh, "Lagrange", 1)         # pressure
W_f = FunctionSpace(mesh, MixedElement([V_f.ufl_element(), Q_f.ufl_element()]))

V_s = VectorFunctionSpace(mesh, "Lagrange", 2)   # displacement
Q_s = FunctionSpace(mesh, "Lagrange", 1)         # solid pressure
W_s = FunctionSpace(mesh, MixedElement([V_s.ufl_element(), Q_s.ufl_element()]))

# -------------------------------------------------
# 3. Trial / test functions & solution Functions
# -------------------------------------------------
(u_f, p_f) = TrialFunctions(W_f)
(v_f, q_f) = TestFunctions(W_f)
w_f   = Function(W_f)          # new fluid state
w_f_n = Function(W_f)          # previous fluid state (zero init)

(d_s, p_s) = TrialFunctions(W_s)
(v_s, q_s) = TestFunctions(W_s)
w_s   = Function(W_s)          # new solid state
w_s_n = Function(W_s)          # previous solid state (zero init)

# -------------------------------------------------
# 4. Material parameters (cgs)
# -------------------------------------------------
mu_f   = 0.003 * 0.1          # 0.003 poise → 0.0003 Pa·s
rho_f  = 1.0
rho_s  = 1.1
nu_s   = 0.49
E_s    = 3.0e5                # Pa
mu_s   = E_s/(2.0*(1.0+nu_s))
lmbda_s= E_s*nu_s/((1.0+nu_s)*(1.0-2.0*nu_s))
K_s    = E_s/(3.0*(1.0-2.0*nu_s))

dt = 1.0e-4
T_end = 0.1

# -------------------------------------------------
# 5. Constitutive relations
# -------------------------------------------------
I = Identity(2)

def sigma_f(u, p):
    return -p*I + 2.0*mu_f*sym(grad(u))

def eps_s(d):
    return sym(grad(d))

def sigma_s(d, p):
    return 2.0*mu_s*eps_s(d) + lmbda_s*tr(eps_s(d))*I - p*I

# -------------------------------------------------
# 6. Bilinear / linear forms (backward Euler)
# -------------------------------------------------
a_f = rho_f/dt*inner(u_f, v_f)*dx_f(1) \
      + inner(sigma_f(u_f, p_f), grad(v_f))*dx_f(1) \
      - q_f*div(u_f)*dx_f(1) \
      - p_f*div(v_f)*dx_f(1)

L_f = rho_f/dt*inner(w_f_n.sub(0), v_f)*dx_f(1)   # modified each step

a_s = rho_s/dt*inner(d_s, v_s)*dx_s(2) \
      + inner(sigma_s(d_s, p_s), grad(v_s))*dx_s(2) \
      - q_s*(tr(eps_s(d_s)) - p_s/K_s)*dx_s(2)

L_s = rho_s/dt*inner(w_s_n.sub(0), v_s)*dx_s(2)   # modified each step

# -------------------------------------------------
# 7. Boundary conditions
# -------------------------------------------------
# Anchor point (lower‑left corner) – eliminates rigid body motions
class Corner(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and near(x[1], -t_wall)

corner = Corner()
bc_s_disp = DirichletBC(W_s.sub(0), Constant((0.0, 0.0)), corner, method="pointwise")

# Fluid: fix pressure at a point (nullspace) and velocity at the same corner
bc_f_p = DirichletBC(W_f.sub(1), Constant(0.0), corner, method="pointwise")
bc_f_u = DirichletBC(W_f.sub(0), Constant((0.0, 0.0)), corner, method="pointwise")
bcs_f = [bc_f_p, bc_f_u]
bcs_s = [bc_s_disp]

# -------------------------------------------------
# 8. XDMF output
# -------------------------------------------------
xdmf_u = XDMFFile(mesh.mpi_comm(), "fluid_velocity.xdmf")
xdmf_d = XDMFFile(mesh.mpi_comm(), "solid_displacement.xdmf")
xdmf_u.parameters["flush_output"] = True
xdmf_d.parameters["flush_output"] = True

# -------------------------------------------------
# 9. Time stepping
# -------------------------------------------------
while t < T_end + DOLFIN_EPS:
    t += dt

    # Time‑dependent inlet traction
    if t <= 5.0e-3:
        T_in = -(2.0e4)/2.0 * (1.0 - cos(pi*t/(2.5e-3)))   # dyn·cm⁻²
        traction = Constant((T_in, 0.0))
    else:
        traction = Constant((0.0, 0.0))

    # Fluid RHS (add inlet traction)
    L_f = rho_f/dt*inner(w_f_n.sub(0), v_f)*dx_f(1) + dot(traction, v_f)*ds(1)

    # Assemble & solve fluid system (keep diagonal for pointwise BCs)
    A_f = assemble(a_f, keep_diagonal=True)
    b_f = assemble(L_f)
    for bc in bcs_f: bc.apply(A_f, b_f)
    solve(A_f, w_f.vector(), b_f, "mumps")

    # Fluid traction on solid walls (bottom & top)
    n_f = FacetNormal(mesh)
    traction_f = sigma_f(w_f.sub(0), w_f.sub(1))*n_f

    # Solid RHS (add fluid traction)
    L_s = rho_s/dt*inner(w_s_n.sub(0), v_s)*dx_s(2) + dot(traction_f, v_s)*(ds(3) + ds(4))

    # Assemble & solve solid system
    A_s = assemble(a_s, keep_diagonal=True)
    b_s = assemble(L_s)
    for bc in bcs_s: bc.apply(A_s, b_s)
    solve(A_s == b_s, w_s, bcs_s, solver_parameters={"linear_solver":"mumps"})

    # Update mesh (solid displacement = mesh displacement)
    ALE.move(mesh, w_s.sub(0))

    # Update previous states
    w_f_n.assign(w_f)
    w_s_n.assign(w_s)

    # Output at requested times
    if near(t, 0.005, dt/2) or near(t, 0.1, dt/2):
        xdmf_u.write(w_f.sub(0), t)   # fluid velocity
        xdmf_d.write(w_s.sub(0), t)   # solid displacement

xdmf_u.close()
xdmf_d.close()