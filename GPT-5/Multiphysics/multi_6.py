from __future__ import print_function
from dolfin import *
from mshr import *
import numpy as np

# --------------------------------------------------------------------------------------
# Parameters (SI)
# --------------------------------------------------------------------------------------
H      = 0.41        # channel height (m)
L      = 2.50        # channel length (m)
xc, yc = 0.20, 0.20  # pole centre (m)
Rc     = 0.05        # pole radius (m)

# Solid (flag) in reference configuration:
x0, y0 = 0.60, 0.19  # right-bottom corner of flag (m)
Lf     = 0.35        # flag length (m)
Tf     = 0.02        # flag thickness (m)

# Fluid
rho_f  = 1000.0           # kg/m^3
nu_f   = 1.0e-3           # m^2/s
mu_f   = rho_f*nu_f       # dynamic viscosity (Pa·s)

# Solid (SVK)
rho_s  = 10000.0          # kg/m^3
nu_s   = 0.4              # Poisson ratio
mu_s   = 0.5e6            # shear modulus (Pa)
lam_s  = 2.0*mu_s*nu_s/(1.0-2.0*nu_s)  # Lamé lambda

# Time stepping
dt      = 0.002
t_end   = 4.0
theta   = 1.0             # backward Euler for fluid
betaNew = 0.25            # Newmark-beta (average acceleration) for solid
gammaNew= 0.5

# Mesh resolutions
res_ch   = 100            # channel resolution
res_flag = 60             # extra along the flag
res_pole = 64

# --------------------------------------------------------------------------------------
# Geometry and mesh (single mesh with subdomains; pole removed)
# --------------------------------------------------------------------------------------
channel = Rectangle(Point(0.0, 0.0), Point(L, H))
pole    = Circle(Point(xc, yc), Rc, res_pole)

# Build flag rectangle in reference configuration
flag    = Rectangle(Point(x0-Lf, y0), Point(x0, y0+Tf))

# Computational domain: channel \ pole    (flag is *not removed*, it is part of domain)
# We keep a single mesh but mark subdomains for fluid and solid.
domain = channel - pole

print("Generating mesh... (this may take a few seconds)")
mesh = generate_mesh(domain, res_ch)

# --------------------------------------------------------------------------------------
# Mark subdomains (fluid=1, solid=2) and facets for BCs
# --------------------------------------------------------------------------------------
cells_marker = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
facets_marker = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)

# Identify solid region by point test
class SolidDomain(SubDomain):
    def inside(self, x, on_boundary):
        return (between(x[0], (x0-Lf - 1e-12, x0 + 1e-12)) and
                between(x[1], (y0 - 1e-12, y0+Tf + 1e-12)))
solid_domain = SolidDomain()

# First set everything to fluid (1), then override to solid (2)
cells_marker.set_all(1)
for cell in cells(mesh):
    cc = cell.midpoint()
    if solid_domain.inside(cc.array(), False):
        cells_marker[cell] = 2

# Boundary markers
INLET, OUTLET, WALLS, POLE, IFACE = 1, 2, 3, 4, 5

class Inlet(SubDomain):
    def inside(self, x, on_bnd): return near(x[0], 0.0) and on_bnd
class Outlet(SubDomain):
    def inside(self, x, on_bnd): return near(x[0], L) and on_bnd
class Walls(SubDomain):
    def inside(self, x, on_bnd): return (near(x[1], 0.0) or near(x[1], H)) and on_bnd
class Pole(SubDomain):
    def inside(self, x, on_bnd):
        return on_bnd and ( (x[0]-xc)**2 + (x[1]-yc)**2 <= (Rc+1e-8)**2 )

Inlet().mark(facets_marker, INLET)
Outlet().mark(facets_marker, OUTLET)
Walls().mark(facets_marker, WALLS)
Pole().mark(facets_marker, POLE)

# Mark the fluid–solid interface (interior facet set) by detecting facets with
# adjacent cells of different subdomain ids.
mesh.init(mesh.topology().dim()-1, mesh.topology().dim())
for f in facets(mesh):
    if f.exterior():  # skip exterior boundary
        continue
    cells_adj = [c for c in cells(f)]
    if len(cells_adj) == 2:
        s0 = cells_marker[cells_adj[0]]
        s1 = cells_marker[cells_adj[1]]
        if (s0, s1) in [(1,2),(2,1)]:
            facets_marker[f] = IFACE

dx = Measure("dx", domain=mesh, subdomain_data=cells_marker)
ds = Measure("ds", domain=mesh, subdomain_data=facets_marker)
dS = Measure("dS", domain=mesh, subdomain_data=facets_marker)  # interior facets

# --------------------------------------------------------------------------------------
# Function spaces
# --------------------------------------------------------------------------------------
Vvec = VectorElement("P", mesh.ufl_cell(), 2)
Vsc  = FiniteElement("P", mesh.ufl_cell(), 1)

# Split by subdomain with a single mixed space over whole mesh:
W_u  = FunctionSpace(mesh, Vvec)  # fluid velocity (whole mesh but used on fluid dx(1))
W_p  = FunctionSpace(mesh, Vsc)   # fluid pressure
W_d  = FunctionSpace(mesh, Vvec)  # solid displacement (used on dx(2))
W_w  = FunctionSpace(mesh, Vvec)  # ALE mesh displacement on fluid to move mesh

# Mixed for fluid solve
W_fluid = MixedFunctionSpace([W_u, W_p])

# --------------------------------------------------------------------------------------
# Boundary and inlet profile
# --------------------------------------------------------------------------------------
Ubar = 1.0
Hc   = H

# steady laminar parabola at t=0
class InletParabola(UserExpression):
    def __init__(self, t, **kwargs):
        self.t = t
        super().__init__(**kwargs)
    def eval(self, values, x):
        y = x[1]
        u0 = 1.5*Ubar*(y*(Hc - y))/((Hc/2.0)**2)
        if self.t < 2.0:
            scale = 0.5*(1.0 - np.cos(np.pi*self.t/2.0))
        else:
            scale = 1.0
        values[0] = scale*u0
        values[1] = 0.0
    def value_shape(self):
        return (2,)

# Solid support (pole side of the flag) is clamped: Γ_s is where the flag meets the pole-hole.
# In our geometry, we simply **fix the solid displacement on the part of the flag that lies on the circle**.
class SolidClamp(SubDomain):
    def inside(self, x, on_bnd):
        on_flag = solid_domain.inside(x, False)
        on_pole_edge = abs((x[0]-xc)**2 + (x[1]-yc)**2 - Rc**2) < 2e-3
        return on_bnd and on_flag and on_pole_edge

# No-slip walls and no-slip on Γ_f (fluid side of the pole)
noslip = Constant((0.0, 0.0))

# --------------------------------------------------------------------------------------
# Unknowns, previous steps and helpers
# --------------------------------------------------------------------------------------
u_p, p_p = Function(W_u), Function(W_p)      # previous fluid solution (for initial guess/convection)
d_n  = Function(W_d)                          # solid disp at t^n
v_n  = Function(W_d)                          # solid vel  at t^n
a_n  = Function(W_d)                          # solid acc  at t^n

w = Function(W_w)                             # ALE mesh disp on fluid
w_dot = Function(W_w)                         # ALE mesh velocity = (w - w_old)/dt
w_old = Function(W_w)

# For output
xf = XDMFFile(mesh.mpi_comm(), "results_fsi/velocity.xdmf")
xp = XDMFFile(mesh.mpi_comm(), "results_fsi/pressure.xdmf")
xd = XDMFFile(mesh.mpi_comm(), "results_fsi/solid_disp.xdmf")
for f in (xf, xp, xd):
    f.parameters["flush_output"] = True
    f.parameters["functions_share_mesh"] = True

# Point A in reference configuration
A_ref = Point(0.60, 0.20)

# --------------------------------------------------------------------------------------
# Solid step: SVK, Newmark in time, solved on subdomain dx(2)
# --------------------------------------------------------------------------------------
d  = Function(W_d)   # unknown at n+1
phi = TestFunction(W_d)

# Newmark prediction
def newmark_predict(dn, vn, an, dt):
    d_pred = project(dn + dt*vn + 0.5*dt*dt*(1.0-2.0*betaNew)*an, W_d)
    v_pred = project(vn + (1.0-gammaNew)*dt*an, W_d)
    return d_pred, v_pred

d_pred, v_pred = newmark_predict(d_n, v_n, a_n, dt)

# Kinematics
I = Identity(2)
F  = I + grad(d)                    # deformation gradient
C  = F.T*F
E  = 0.5*(C - I)                    # Green-Lagrange
S  = lam_s*tr(E)*I + 2.0*mu_s*E     # 2nd PK
P_s = F*S                           # 1st PK

# Effective inertia for Newmark (consistent mass, lumped via rho_s)
a = (d - d_pred)*(1.0/(betaNew*dt*dt))
v = v_pred + gammaNew*dt*a

# Weak form on solid only
Res_solid = ( rho_s*inner(a, phi)*dx(2)
              + inner(P_s, grad(phi))*dx(2) )

d_solid_problem = NonlinearVariationalProblem(Res_solid, d)
d_solid_solver  = NonlinearVariationalSolver(d_solid_problem)
prm = d_solid_solver.parameters
prm["newton_solver"]["absolute_tolerance"] = 1e-8
prm["newton_solver"]["relative_tolerance"] = 1e-7
prm["newton_solver"]["maximum_iterations"] = 25
prm["newton_solver"]["report"] = True
prm["newton_solver"]["linear_solver"] = "mumps"

# Clamp Γ_s: zero displacement on the part of the flag touching the pole rim
bc_s_list = []

# We approximate Γ_s as the intersection of the solid boundary with the pole boundary.
# Build a DirichletBC using a compiled SubDomain that filters to solid cells.
class GammaS(SubDomain):
    def inside(self, x, on_bnd):
        # on the circle and inside the flag rectangle
        on_circle = near((x[0]-xc)**2 + (x[1]-yc)**2, Rc**2, 5e-3)
        in_flag = (x0-Lf - 1e-12 <= x[0] <= x0 + 1e-12) and (y0 - 1e-12 <= x[1] <= y0+Tf + 1e-12)
        return on_bnd and on_circle and in_flag

gammaS = GammaS()
bc_s_list.append(DirichletBC(W_d, Constant((0.0, 0.0)), gammaS))

# --------------------------------------------------------------------------------------
# ALE mesh motion on the fluid: harmonic extension, w = d on interface, w = 0 on boundaries
# --------------------------------------------------------------------------------------
psi = TestFunction(W_w)
w_trial = TrialFunction(W_w)

a_ale = inner(grad(w_trial), grad(psi))*dx(1)
L_ale = Constant((0.0, 0.0))*psi[0]*dx(1)  # zero RHS

# Boundary conditions: w=0 on inlet/outlet/walls and on rigid pole boundary; w=d on Γ_fs
bcs_ale = [
    DirichletBC(W_w, Constant((0.0, 0.0)), facets_marker, INLET),
    DirichletBC(W_w, Constant((0.0, 0.0)), facets_marker, OUTLET),
    DirichletBC(W_w, Constant((0.0, 0.0)), facets_marker, WALLS),
    DirichletBC(W_w, Constant((0.0, 0.0)), facets_marker, POLE),
]

# For the fluid–solid interface (interior), impose w=d strongly via a boundary mask:
# Build a Function to hold interface values by restriction later (we apply after solid solve).
# In legacy FEniCS we cannot DirichletBC on interior facets directly, so we enforce by
# overwriting dofs whose support intersects IFACE using a proximity criterion.
iface_dofs = []
V2dof = W_w.dofmap()
coords = W_w.tabulate_dof_coordinates().reshape((-1, 2))
for dof, x in enumerate(coords):
    # Close to the interface facets (cheap proximity test):
    on_flag = solid_domain.inside(x, False)
    # pick nodes within a narrow band around the rectangle edge except the clamped edge
    near_left_edge  = near(x[0], x0-Lf, 1.5*mesh.hmin())
    near_right_edge = near(x[0], x0,     1.5*mesh.hmin())
    near_bottom     = near(x[1], y0,     1.5*mesh.hmin())
    near_top        = near(x[1], y0+Tf,  1.5*mesh.hmin())
    # the true interface is the outer boundary of the flag (except where clamped on the pole)
    if (near_left_edge or near_right_edge or near_bottom or near_top) and not on_flag:
        iface_dofs.append(dof)
iface_dofs = list(set(iface_dofs))

A_ale = assemble(a_ale)
b_ale = None  # built every step because w=d appears in bc handling

# --------------------------------------------------------------------------------------
# Fluid step: ALE Navier–Stokes (incremental pressure-correction style for robustness)
# --------------------------------------------------------------------------------------
(u, p)   = TrialFunctions(W_fluid)
(v, q)   = TestFunctions(W_fluid)

u_n = Function(W_u)  # last converged
p_n = Function(W_p)

def mesh_velocity(w, w_old, dt):
    wdot = Function(W_w)
    wdot.vector()[:] = (w.vector() - w_old.vector())/dt
    return wdot

def conv(u, wdot):
    return dot((u - wdot), nabla_grad(u))

# Mass and momentum forms (θ-scheme)
u_mid = u  # backward Euler (theta=1)
wdot = Function(W_w)  # filled each step

a_fluid = ( rho_f/dt*inner(u, v)*dx(1)
            + rho_f*inner(dot((u_p - wdot), nabla_grad(u)), v)*dx(1)
            + 2.0*mu_f*inner(sym(grad(u)), sym(grad(v)))*dx(1)
            - div(v)*p*dx(1)
            - q*div(u)*dx(1) )

L_fluid = ( rho_f/dt*inner(u_n, v)*dx(1) )

# Inlet BC for velocity; no-slip on walls and pole; interface no-slip set to solid velocity
bcs_fluid_u = [
    DirichletBC(W_fluid.sub(0), noslip, facets_marker, WALLS),
    DirichletBC(W_fluid.sub(0), noslip, facets_marker, POLE),
]
# Inlet parabola is time-dependent; added inside loop.
# Outlet: traction-free → no Dirichlet for u; impose p=0 reference (weakly) or do nothing (natural).
# We fix mean pressure to zero each step to avoid singularity.
null_p = Constant(0.0)

# --------------------------------------------------------------------------------------
# Time integration
# --------------------------------------------------------------------------------------
t = 0.0
step = 0
A_data = []  # [(t, dx, dy)]

# Helper: write outputs
def write_outputs(t):
    xf.write(u_p, t)
    xp.write(p_p, t)
    xd.write(d_n, t)

# Interpolate initial fields
u_p.assign(Constant((0.0, 0.0)))
p_p.assign(Constant(0.0))
u_n.assign(u_p)
p_n.assign(p_p)
d_n.assign(Constant((0.0, 0.0)))
v_n.assign(Constant((0.0, 0.0)))
a_n.assign(Constant((0.0, 0.0)))
w.vector()[:] = 0.0
w_old.vector()[:] = 0.0

# Ensure output directory exists (FEniCS creates lazily via XDMF)

# Main loop
while t < t_end + 1e-12:
    t += dt
    step += 1
    print("\n=== Time step %d  t = %.4f ===" % (step, t))

    # ---------------------
    # (1) Solid (SVK) step
    # ---------------------
    # update prediction with present d_n, v_n, a_n
    d_pred, v_pred = newmark_predict(d_n, v_n, a_n, dt)

    # Rebuild solid residual with updated predictors (they enter via a, v definitions)
    # Recreate problem to refresh closures:
    a = (d - d_pred)*(1.0/(betaNew*dt*dt))
    v = v_pred + gammaNew*dt*a
    F  = I + grad(d)
    C  = F.T*F
    E  = 0.5*(C - I)
    S  = lam_s*tr(E)*I + 2.0*mu_s*E
    P_s = F*S
    Res_solid = ( rho_s*inner(a, phi)*dx(2) + inner(P_s, grad(phi))*dx(2) )
    d_solid_problem = NonlinearVariationalProblem(Res_solid, d, bc_s_list)
    d_solid_solver  = NonlinearVariationalSolver(d_solid_problem)
    prm = d_solid_solver.parameters
    prm["newton_solver"]["absolute_tolerance"] = 1e-8
    prm["newton_solver"]["relative_tolerance"] = 1e-7
    prm["newton_solver"]["maximum_iterations"] = 25
    prm["newton_solver"]["linear_solver"] = "mumps"
    d_solid_solver.solve()

    # Update solid kinematics
    a_new = project((d - d_pred)*(1.0/(betaNew*dt*dt)), W_d)
    v_new = project(v_pred + gammaNew*dt*a_new, W_d)

    # ---------------------
    # (2) ALE mesh motion
    # ---------------------
    # Assemble RHS and apply interface Dirichlet data by dof injection: w = d on interface
    b_ale = assemble(inner(Constant((0.0,0.0)), psi)*dx(1))
    # Start with homogeneous bcs
    for bc in bcs_ale:
        bc.apply(A_ale, b_ale)
    # Enforce interface values: copy solid displacement onto nearby fluid dofs
    w_vec = w.vector()[:]  # start with previous
    d_vec = d.vector()
    # Build a KDTree would be nicer; here, simply evaluate d at dof coordinates:
    d_eval = d(dolfin.FunctionSpace(mesh, Vvec).tabulate_dof_coordinates()[0:1]) if False else None
    d_at_nodes = d # callable
    for dof in iface_dofs:
        x = coords[dof]
        w_vec[2*dof+0] = d(x)[0]
        w_vec[2*dof+1] = d(x)[1]
    w.vector()[:] = w_vec
    # Solve harmonic extension for the interior fluid nodes
    solve(A_ale, w.vector(), b_ale, "cg", "ilu")

    # Mesh velocity
    w_dot.assign(mesh_velocity(w, w_old, dt))

    # Move the mesh to the new configuration: X_new = X + w   (only for fluid subdomain)
    # For robustness we move the entire mesh; solid displacement already represented by d on dx(2).
    ALE.move(mesh, w)
    mesh.bounding_box_tree().build(mesh)

    # ---------------------
    # (3) Fluid step on updated mesh (backward Euler ALE NS)
    # ---------------------
    inlet_profile = InletParabola(t, degree=2)
    bc_inlet = DirichletBC(W_fluid.sub(0), inlet_profile, facets_marker, INLET)
    # Interface no-slip: enforce u = v_solid on the flag boundary seen from the fluid
    # We reuse the same iface_dofs logic by constructing a pointwise BC.
    u_if = Function(W_u)
    u_if_vec = u_if.vector()[:]
    for dof in iface_dofs:
        x = coords[dof]
        u_if_vec[2*dof+0] = v_new(x)[0]
        u_if_vec[2*dof+1] = v_new(x)[1]
    u_if.vector()[:] = u_if_vec

    # Build system
    (u_test, p_test) = TrialFunctions(W_fluid)
    (vF, qF)         = TestFunctions(W_fluid)

    a_fluid = ( rho_f/dt*inner(u_test, vF)*dx(1)
                + rho_f*inner(dot((u_p - w_dot), nabla_grad(u_test)), vF)*dx(1)
                + 2.0*mu_f*inner(sym(grad(u_test)), sym(grad(vF)))*dx(1)
                - div(vF)*p_test*dx(1)
                - qF*div(u_test)*dx(1) )
    L_fluid = ( rho_f/dt*inner(u_p, vF)*dx(1) )

    # Assemble and apply BCs
    U_sol = Function(W_fluid)
    A_fl = assemble(a_fluid)
    b_fl = assemble(L_fluid)

    # Standard BCs
    for bc in [bc_inlet] + bcs_fluid_u:
        bc.apply(A_fl, b_fl)

    # Interface no-slip by strong imposition at velocity dofs (pressure untouched)
    # We project u_if onto the mixed vector block:
    u_submap = W_fluid.sub(0).dofmap()
    U_vec = U_sol.vector()
    # Solve linear system
    solve(A_fl, U_vec, b_fl, "gmres", "ilu")
    # Overwrite velocity dofs near interface with v_new values
    # (keeps the pressure solution intact)
    U_local = U_vec.get_local()
    Ui_dofs = u_submap.dofs()
    u_curr  = Function(W_u)
    assigner = FunctionAssigner(W_u, W_fluid.sub(0))
    assigner.assign(u_curr, U_sol.sub(0))
    u_curr_vec = u_curr.vector()[:]
    for dof in iface_dofs:
        x = coords[dof]
        u_curr_vec[2*dof+0] = v_new(x)[0]
        u_curr_vec[2*dof+1] = v_new(x)[1]
    u_curr.vector()[:] = u_curr_vec
    # write back into mixed vector
    assigner_back = FunctionAssigner(W_fluid.sub(0), W_u)
    assigner_back.assign(U_sol.sub(0), u_curr)

    # Split for output
    u_p.assign(U_sol.sub(0))
    p_p.assign(U_sol.sub(1))

    # ---------------------
    # (4) Update histories and record outputs
    # ---------------------
    w_old.assign(w)
    u_n.assign(u_p)
    p_n.assign(p_p)
    d_n.assign(d)
    v_n.assign(v_new)
    a_n.assign(a_new)

    # Displacement of point A (solid reference point)
    # Evaluate safely by mapping to current location: since solid is Lagrangian, d_n(A_ref)
    try:
        dA = d_n(A_ref)
        A_data.append((t, dA[0], dA[1]))
        if MPI.rank(mesh.mpi_comm()) == 0:
            print("A(t): dx = %.6e, dy = %.6e" % (dA[0], dA[1]))
    except RuntimeError:
        if MPI.rank(mesh.mpi_comm()) == 0:
            print("Warning: point A outside solid interpolation patch at this mesh resolution.")

    write_outputs(t)

print("\nSimulation complete.")
if MPI.rank(mesh.mpi_comm()) == 0:
    # Save A(t) as CSV
    with open("results_fsi/A_displacement.csv", "w") as f:
        f.write("t,dx,dy\n")
        for (tt, dxA, dyA) in A_data:
            f.write("{:.8f},{:.8e},{:.8e}\n".format(tt, dxA, dyA))
    print("Saved: results_fsi/A_displacement.csv")
