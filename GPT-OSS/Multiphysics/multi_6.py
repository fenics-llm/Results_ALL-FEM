# filename: fsi_flag.py
# --------------------------------------------------------------
# 2‑D Fluid–Structure Interaction: flag in a channel (legacy FEniCS)
# --------------------------------------------------------------
#   Author:  ChatGPT (based on the FEniCS “fsi.py” demo)
#   Date:    2025‑12‑02
# --------------------------------------------------------------

from __future__ import print_function
import numpy as np
import os
from dolfin import *
from mshr import *

# ----------------------------------------------------------------------
# 1. PARAMETERS
# ----------------------------------------------------------------------
# Geometry
L = 2.5          # channel length  [m]
H = 0.41         # channel height  [m]

# Flag (solid) rectangle
flag_len   = 0.35
flag_thick = 0.02
flag_bl    = Point(0.60, 0.19)          # lower‑left corner of the flag

# Pole (rigid disk) – removed from the computational domain
pole_center = Point(0.20, 0.20)
pole_radius = 0.05

# Physical parameters
rho_f = 1000.0          # fluid density   [kg/m³]
nu_f  = 1.0e-3          # kinematic viscosity [m²/s]
mu_f  = rho_f*nu_f      # dynamic viscosity

rho_s = 10000.0         # solid density   [kg/m³]
nu_s  = 0.4             # Poisson ratio
mu_s  = 0.5e6           # shear modulus   [Pa]
lambda_s = 2*mu_s*nu_s/(1-2*nu_s)   # Lame parameter for SVK

# Time stepping
T  = 8.0               # final time   [s]
dt = 0.02              # time step    [s]
num_steps = int(T/dt)

# Inlet profile (parabolic)
Ubar = 1.0
def inlet_parabola(y):
    return 1.5*Ubar*y*(H-y)/(H/2.0)**2

# ----------------------------------------------------------------------
# 2. MESH (single mesh for fluid + solid)
# ----------------------------------------------------------------------
channel = Rectangle(Point(0.0, 0.0), Point(L, H))
pole    = Circle(pole_center, pole_radius, 64)

# Whole computational domain = channel minus pole
domain = channel - pole
mesh = generate_mesh(domain, 80)          # 80 ≈ mesh resolution

# ----------------------------------------------------------------------
# 3. SUB‑DOMAIN MARKERS (fluid = 0, solid = 1)
# ----------------------------------------------------------------------
class SolidDomain(SubDomain):
    def inside(self, x, on_boundary):
        return (flag_bl.x() <= x[0] <= flag_bl.x()+flag_len and
                flag_bl.y() <= x[1] <= flag_bl.y()+flag_thick)

# Cell markers
subdomains = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
SolidDomain().mark(subdomains, 1)

# ----------------------------------------------------------------------
# 4. FACET MARKERS (for BCs and the fluid–solid interface)
# ----------------------------------------------------------------------
class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and on_boundary
class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], L) and on_boundary
class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[1], 0.0) or near(x[1], H)) and on_boundary
class Interface(SubDomain):
    def inside(self, x, on_boundary):
        # points that belong to the flag boundary and are not on the pole
        on_flag = (near(x[0], flag_bl.x()) or near(x[0], flag_bl.x()+flag_len) or
                  near(x[1], flag_bl.y()) or near(x[1], flag_bl.y()+flag_thick))
        not_pole = ( (x[0]-pole_center.x())**2 + (x[1]-pole_center.y())**2
                     > (pole_radius+ DOLFIN_EPS)**2 )
        return on_flag and not_pole and on_boundary

boundaries = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
Inlet().mark(boundaries, 1)
Outlet().mark(boundaries, 2)
Walls().mark(boundaries, 3)
Interface().mark(boundaries, 4)

# ----------------------------------------------------------------------
# 5. FUNCTION SPACES
# ----------------------------------------------------------------------
V = VectorElement('P', mesh.ufl_cell(), 2)   # velocity / displacement (P2)
Q = FiniteElement('P', mesh.ufl_cell(), 1)  # pressure (P1)

mixed_elem = MixedElement([V, Q, V])        # (u, p, d_s)
W = FunctionSpace(mesh, mixed_elem)

# Separate measures for fluid and solid cells
dx = Measure('dx', domain=mesh, subdomain_data=subdomains)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# ----------------------------------------------------------------------
# 6. TRIAL / TEST FUNCTIONS & SOLUTION FUNCTIONS
# ----------------------------------------------------------------------
w   = Function(W)          # (u, p, d_s) at new time level
w0  = Function(W)          # at previous time level
w1  = Function(W)          # at time level n‑1 (for BDF2)

(u, p, d_s) = split(w)
(u0, p0, d_s0) = split(w0)
(u1, p1, d_s1) = split(w1)

(v, q, eta) = TestFunctions(W)

# Mesh displacement (used for ALE)
V_mesh = VectorFunctionSpace(mesh, 'P', 1)
d_mesh   = Function(V_mesh, name='mesh_disp')
d_mesh0  = Function(V_mesh, name='mesh_disp_old')
w_mesh = (d_mesh - d_mesh0)/dt          # mesh velocity

# ----------------------------------------------------------------------
# 7. KINEMATICS (solid)
# ----------------------------------------------------------------------
I = Identity(2)
F = I + grad(d_s)               # deformation gradient
E = 0.5*(F.T*F - I)            # Green‑Lagrange strain
S = lambda_s*tr(E)*I + 2.0*mu_s*E   # 2nd Piola‑Kirchhoff (SVK)
P = F*S                         # 1st Piola‑Kirchhoff

# Solid acceleration (BDF2)
beta = 0.0
d2s_dt2 = (1.0/(dt*dt))*((2.0+beta)*d_s - (4.0+2.0*beta)*d_s0 + (2.0+beta)*d_s1)

# ----------------------------------------------------------------------
# 8. FLUID EQUATIONS (ALE Navier–Stokes, BDF2)
# ----------------------------------------------------------------------
u_mid = 1.5*u - 0.5*u0          # extrapolation for convection term

F_fluid = (rho_f/dt)*inner(u - u0, v)*dx(0) \
          + rho_f*inner(dot(u_mid - w_mesh, grad(u_mid)), v)*dx(0) \
          + mu_f*inner(grad(u), grad(v))*dx(0) \
          - div(v)*p*dx(0) \
          + q*div(u)*dx(0)

# ----------------------------------------------------------------------
# 9. SOLID EQUATIONS (dynamic elasticity)
# ----------------------------------------------------------------------
F_solid = rho_s*inner(d2s_dt2, eta)*dx(1) + inner(P, grad(eta))*dx(1)

# ----------------------------------------------------------------------
# 10. COUPLING (Nitsche) – velocity continuity & traction balance
# ----------------------------------------------------------------------
gamma = Constant(10.0*mu_f/dt)          # penalty parameter
n = FacetNormal(mesh)

# u = ḋ_s on the interface (velocity continuity) + traction equilibrium
F_couple = gamma*inner(u - (d_s - d_s0)/dt, v)*ds(4) \
           - inner(mu_f*grad(u)*n, v)*ds(4) \
           - inner(mu_f*grad(v)*n, u - (d_s - d_s0)/dt)*ds(4)

# ----------------------------------------------------------------------
# 11. TOTAL RESIDUAL
# ----------------------------------------------------------------------
F = F_fluid + F_solid + F_couple

# ----------------------------------------------------------------------
# 12. BOUNDARY CONDITIONS
# ----------------------------------------------------------------------
bcs = []

# Inlet – time‑dependent parabolic profile
inlet_expr = Expression(('Umax*(1.0 - cos(pi*t/2.0))/2.0',
                         '0.0'), degree=2,
                         Umax=Constant(0.0), t=0.0, pi=np.pi)
bcs.append(DirichletBC(W.sub(0), inlet_expr, boundaries, 1))

# No‑slip on top & bottom walls
zero_vec = Constant((0.0, 0.0))
bcs.append(DirichletBC(W.sub(0), zero_vec, boundaries, 3))

# Fixed part of the solid (the left edge of the flag)
bcs.append(DirichletBC(W.sub(2), Constant((0.0, 0.0)), boundaries, 1))   # facet 1 = inlet = also the left edge of flag (coincident)

# ----------------------------------------------------------------------
# 13. SOLVER SETTINGS
# ----------------------------------------------------------------------
problem = NonlinearVariationalProblem(F, w, bcs)
solver  = NonlinearVariationalSolver(problem)

prm = solver.parameters
prm['newton_solver']['absolute_tolerance'] = 1e-8
prm['newton_solver']['relative_tolerance'] = 1e-6
prm['newton_solver']['maximum_iterations'] = 25
prm['newton_solver']['linear_solver'] = 'mumps'

# ----------------------------------------------------------------------
# 14. OUTPUT (XDMF) + POINT‑A TRACKING
# ----------------------------------------------------------------------
os.makedirs('results', exist_ok=True)

xdmf_u = XDMFFile('results/velocity.xdmf')
xdmf_p = XDMFFile('results/pressure.xdmf')
xdmf_d = XDMFFile('results/solid_disp.xdmf')
xdmf_u.parameters["flush_output"] = True
xdmf_p.parameters["flush_output"] = True
xdmf_d.parameters["flush_output"] = True

csv_file = open('results/pointA_disp.csv', 'w')
csv_file.write('t,dx,dy\n')
point_A = Point(0.60, 0.20)   # reference location of point A

def write_point_A(t, d_s_func):
    dA = d_s_func(point_A)
    csv_file.write('{:.5f},{:.8e},{:.8e}\n'.format(t, dA[0], dA[1]))

# ----------------------------------------------------------------------
# 15. TIME‑STEPPING LOOP
# ----------------------------------------------------------------------
t = 0.0
for n in range(num_steps):
    t += dt
    print('--- time step %d / %d, t = %.3f ---' % (n+1, num_steps, t))

    # Update inlet expression (maximum velocity of the parabola)
    Umax = 1.5*Ubar*H/4.0
    inlet_expr.Umax = Umax
    inlet_expr.t    = t

    # --------------------------------------------------------------
    # 15.1  Mesh motion (Laplace problem for ALE)
    # --------------------------------------------------------------
    V_mesh_trial = TrialFunction(V_mesh)
    v_mesh_test  = TestFunction(V_mesh)

    a_mesh = inner(grad(V_mesh_trial), grad(v_mesh_test))*dx
    L_mesh = Constant(0.0)*v_mesh_test*dx

    # Dirichlet BCs for mesh motion:
    bc_mesh = []
    # Zero displacement on inlet, outlet, walls
    bc_mesh.append(DirichletBC(V_mesh, Constant((0.0, 0.0)), boundaries, 1))
    bc_mesh.append(DirichletBC(V_mesh, Constant((0.0, 0.0)), boundaries, 2))
    bc_mesh.append(DirichletBC(V_mesh, Constant((0.0, 0.0)), boundaries, 3))

    # On the fluid–solid interface we prescribe the solid displacement
    # (project solid displacement onto the mesh space)
    d_s_func = Function(V)          # full mixed space, we only need the displacement component
    d_s_func.vector()[:] = w.sub(2).vector()
    d_s_proj = project(d_s_func, V_mesh)
    bc_mesh.append(DirichletBC(V_mesh, d_s_proj, boundaries, 4))

    solve(a_mesh == L_mesh, d_mesh, bc_mesh)

    # --------------------------------------------------------------
    # 15.2  ALE mapping (move fluid mesh)
    # --------------------------------------------------------------
    ALE.move(mesh, d_mesh - d_mesh0)
    d_mesh0.assign(d_mesh)

    # --------------------------------------------------------------
    # 15.3  Solve the monolithic FSI system
    # --------------------------------------------------------------
    solver.solve()

    # --------------------------------------------------------------
    # 15.4  Save results
    # --------------------------------------------------------------
    (u_sol, p_sol, d_s_sol) = w.split()
    xdmf_u.write(u_sol, t)
    xdmf_p.write(p_sol, t)
    xdmf_d.write(d_s_sol, t)

    # --------------------------------------------------------------
    # 15.5  Record displacement of point A
    # --------------------------------------------------------------
    write_point_A(t, d_s_sol)

    # --------------------------------------------------------------
    # 15.6  Rotate solution vectors for BDF2
    # --------------------------------------------------------------
    w1.assign(w0)
    w0.assign(w)

print('Simulation finished.')
csv_file.close()