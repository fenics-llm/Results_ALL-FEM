# filename: fsi_2d_elastic_tube.py
from dolfin import *
import numpy as np

# -------------------------------------------------
#  Parameters (CGS units: cm, g, s)
# -------------------------------------------------
L  = 6.0          # tube length (cm)
Hf = 1.0          # fluid height (cm)
ts = 0.1          # solid wall thickness (cm)

mu_f   = 0.003               # fluid viscosity (poise)
rho_f  = 1.0                 # fluid density (g/cm³)
rho_s  = 1.1                 # solid density (g/cm³)
nu_s   = 0.49                # solid Poisson ratio
E_s    = 3.0e5               # Young modulus (Pa) → dyne/cm² (1 Pa = 10 dyne/cm²)
E_s   *= 10.0                # now in dyne/cm²

# Lame parameters for the solid (cgs)
lambda_s = E_s*nu_s/((1.0+nu_s)*(1.0-2.0*nu_s))
mu_s     = E_s/(2.0*(1.0+nu_s))

dt = 1.0e-4          # time step (s)
T  = 0.1             # final time (s)

# -------------------------------------------------
#  Mesh (fluid + two solid layers)
# -------------------------------------------------
nx = 120                     # cells along x (fluid)
ny_f = 20                    # cells in fluid y‑direction
ny_s = 4                     # cells in each solid layer

mesh = RectangleMesh(Point(0.0, -ts), Point(L, Hf + ts),
                     nx, ny_f + 2*ny_s)

# -------------------------------------------------
#  Subdomains (fluid = 0, solid = 1)
# -------------------------------------------------
fluid_id = 0
solid_id = 1
domains = MeshFunction("size_t", mesh, mesh.topology().dim(), fluid_id)

class FluidDomain(SubDomain):
    def inside(self, x, on_boundary):
        return (x[1] > 0.0) and (x[1] < Hf)

class SolidDomain(SubDomain):
    def inside(self, x, on_boundary):
        return (x[1] <= 0.0) or (x[1] >= Hf)

FluidDomain().mark(domains, fluid_id)
SolidDomain().mark(domains, solid_id)

dx = Measure('dx', domain=mesh, subdomain_data=domains)

# -------------------------------------------------
#  Boundary markers
# -------------------------------------------------
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)

class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and on_boundary

class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], L) and on_boundary

class WallFluid(SubDomain):
    # fluid–solid interface (top & bottom of fluid)
    def inside(self, x, on_boundary):
        return (near(x[1], 0.0) or near(x[1], Hf)) and on_boundary

class OuterWall(SubDomain):
    # outer faces of the solid layers (traction free)
    def inside(self, x, on_boundary):
        return (near(x[1], -ts) or near(x[1], Hf + ts)) and on_boundary

inlet_id, outlet_id, wall_id, outer_id = 1, 2, 3, 4
Inlet().mark(boundaries, inlet_id)
Outlet().mark(boundaries, outlet_id)
WallFluid().mark(boundaries, wall_id)
OuterWall().mark(boundaries, outer_id)

ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# -------------------------------------------------
#  Function spaces (mixed fluid–solid)
# -------------------------------------------------
Vf = VectorElement("P", mesh.ufl_cell(), 2)   # fluid velocity
Qf = FiniteElement("P", mesh.ufl_cell(), 1)   # fluid pressure
Vs = VectorElement("P", mesh.ufl_cell(), 2)   # solid displacement
Qs = FiniteElement("P", mesh.ufl_cell(), 1)   # solid pressure (mixed)

W_elem = MixedElement([Vf, Qf, Vs, Qs])
W = FunctionSpace(mesh, W_elem)

# -------------------------------------------------
#  Unknowns at current and previous time steps
# -------------------------------------------------
w   = Function(W)          # (u_f, p_f, u_s, p_s) at new time
w_n = Function(W)          # at previous time

(vf, qf, vs, qs) = TestFunctions(W)
(uf, pf, us, ps) = split(w)
(uf_n, pf_n, us_n, ps_n) = split(w_n)

# -------------------------------------------------
#  Kinematics & stresses
# -------------------------------------------------
I = Identity(2)

def epsilon(v):
    return sym(grad(v))

# Fluid Cauchy stress
sigma_f = -pf*I + mu_f*(grad(uf) + grad(uf).T)

# Solid stress (mixed formulation, nearly incompressible)
sigma_s = 2.0*mu_s*epsilon(us) - ps*I

# -------------------------------------------------
#  Weak forms
# -------------------------------------------------
# Fluid momentum + continuity (Stokes, no convection for robustness)
F_fluid = (rho_f/dt)*dot(uf - uf_n, vf)*dx(fluid_id) \
          + inner(sigma_f, grad(vf))*dx(fluid_id) \
          - qf*div(uf)*dx(fluid_id)

# Solid momentum + incompressibility (mixed)
F_solid = (rho_s/dt)*dot(us - us_n, vs)*dx(solid_id) \
          + inner(sigma_s, grad(vs))*dx(solid_id) \
          - qs*div(us)*dx(solid_id)

# -------------------------------------------------
#  Boundary conditions
# -------------------------------------------------
# No‑slip on the fluid–solid interface (fluid velocity = 0)
bc_noslip = DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, wall_id)

# Fix solid displacement on the outer faces (to avoid rigid‑body motion)
zero_vec = Constant((0.0, 0.0))
bc_solid_outer = DirichletBC(W.sub(2), zero_vec, boundaries, outer_id)

# Pin one pressure DOF for each mixed pressure to remove the null‑space
# (pointwise Dirichlet on a single vertex)
p0 = Point(0.0, Hf/2.0)          # somewhere inside the fluid
bc_pf = DirichletBC(W.sub(1), Constant(0.0), p0, method='pointwise')
p1 = Point(L, Hf/2.0)           # somewhere inside the solid
bc_ps = DirichletBC(W.sub(3), Constant(0.0), p1, method='pointwise')

bcs = [bc_noslip, bc_solid_outer, bc_pf, bc_ps]

# -------------------------------------------------
#  Time‑dependent inlet traction
# -------------------------------------------------
t = Constant(0.0)                     # physical time (scalar)
traction_mag = Constant(0.0)          # scalar magnitude of traction

# Vector traction = (traction_mag, 0)
traction_vec = as_vector([traction_mag, Constant(0.0)])

# Natural BC for the prescribed traction
F_inlet = dot(sigma_f*FacetNormal(mesh), vf)*ds(inlet_id) - dot(traction_vec, vf)*ds(inlet_id)

# -------------------------------------------------
#  Total residual and Jacobian
# -------------------------------------------------
F = F_fluid + F_solid + F_inlet
J = derivative(F, w)

# -------------------------------------------------
#  Solver configuration (direct solver – robust for this small test)
# -------------------------------------------------
problem = NonlinearVariationalProblem(F, w, bcs, J)
solver  = NonlinearVariationalSolver(problem)
prm = solver.parameters
prm['newton_solver']['absolute_tolerance'] = 1e-8
prm['newton_solver']['relative_tolerance'] = 1e-6
prm['newton_solver']['maximum_iterations'] = 25
prm['newton_solver']['linear_solver'] = 'mumps'   # direct solver

# -------------------------------------------------
#  Output (XDMF)
# -------------------------------------------------
xdmf = XDMFFile(mesh.mpi_comm(), "fsi_results.xdmf")
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True

# -------------------------------------------------
#  Time stepping
# -------------------------------------------------
time = 0.0
while time < T + DOLFIN_EPS:
    time += dt
    t.assign(time)

    # ----- inlet traction (only for t < 0.005 s) -----
    if time < 0.005:
        mag = -(2.0e4)/2.0 * (1.0 - np.cos(np.pi*time/(2.5e-3)))
    else:
        mag = 0.0
    traction_mag.assign(mag)

    # Solve the coupled problem
    solver.solve()

    # Save results at the requested instants
    if near(time, 0.005, dt/2) or near(time, 0.1, dt/2):
        (uf_out, pf_out, us_out, ps_out) = w.split()
        uf_out.rename("fluid_velocity", "velocity")
        us_out.rename("solid_displacement", "displacement")
        xdmf.write(uf_out, time)
        xdmf.write(us_out, time)

    # Prepare for next step
    w_n.assign(w)

xdmf.close()