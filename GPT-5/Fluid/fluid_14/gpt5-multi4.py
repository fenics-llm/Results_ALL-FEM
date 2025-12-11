from __future__ import print_function
from dolfin import *
from mshr import *
import numpy as np

# -----------------------
# Problem parameters
# -----------------------
U_in   = 1.0             # m/s
nu     = 2.56e-5         # kinematic viscosity (m^2/s)
rho    = 1.0             # kg/m^3
D      = 1.0             # m (cylinder diameter)
R      = 0.5*D
Lbox   = 30.0*D          # half-length of square
t_end  = 10.0            # s
dt     = 0.01            # s  (adjust if needed)
theta  = 1.0             # BE/θ-method (1.0 = backward Euler)

# VMS / stabilisation parameters
C_tau  = 1.0             # coefficient in tau_m, commonly O(1)
gamma_gd = 0.1           # grad-div coefficient (adjust if needed)
# You can also tune PSPG/SUPG via multipliers if required
supg_scale  = 1.0
pspg_scale  = 1.0

# Mesh resolution (increase for accuracy)
# Using mshr; you may replace with a pre-generated mesh for serious runs.
resolution = 180  # global resolution for mshr (increase near cylinder for fidelity)

# -----------------------
# Mesh and boundaries
# -----------------------
rect = Rectangle(Point(-Lbox, -Lbox), Point(Lbox, Lbox))
cyl  = Circle(Point(0.0, 0.0), R)
domain = rect - cyl
mesh = generate_mesh(domain, resolution)

# Mark boundaries
LEFT, RIGHT, TOP, BOTTOM, CYL = 1, 2, 3, 4, 5
class Left(SubDomain):
    def inside(self, x, on_boundary):  return on_boundary and near(x[0], -Lbox, DOLFIN_EPS)
class Right(SubDomain):
    def inside(self, x, on_boundary):  return on_boundary and near(x[0],  Lbox, DOLFIN_EPS)
class Top(SubDomain):
    def inside(self, x, on_boundary):  return on_boundary and near(x[1],  Lbox, DOLFIN_EPS)
class Bottom(SubDomain):
    def inside(self, x, on_boundary):  return on_boundary and near(x[1], -Lbox, DOLFIN_EPS)
class Cylinder(SubDomain):
    def inside(self, x, on_boundary):  # circle centre (0,0), radius R
        return on_boundary and near((x[0]**2 + x[1]**2)**0.5, R, 1e-8)

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
Left().mark(boundaries, LEFT)
Right().mark(boundaries, RIGHT)
Top().mark(boundaries, TOP)
Bottom().mark(boundaries, BOTTOM)
Cylinder().mark(boundaries, CYL)
ds_ = Measure("ds", domain=mesh, subdomain_data=boundaries)

n = FacetNormal(mesh)

# -----------------------
# FE spaces
# -----------------------
V = VectorFunctionSpace(mesh, "Lagrange", 1)
Q = FunctionSpace(mesh, "Lagrange", 1)
W = MixedFunctionSpace([V, Q])

(u, p)   = TrialFunctions(W)
(v, q)   = TestFunctions(W)

w_   = Function(W)     # current unknown (u, p)
w_n  = Function(W)     # previous step (u_n, p_n)
u_, p_ = split(w_)
u_n, p_n = split(w_n)

# -----------------------
# Boundary conditions
# -----------------------
# Inflow: u=(U,0) at x=-Lbox
inlet_profile = Constant((U_in, 0.0))

bcs = []
bcs.append(DirichletBC(W.sub(0), inlet_profile, boundaries, LEFT))   # velocity at inlet
# No-slip on top, bottom, cylinder
bcs.append(DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, TOP))
bcs.append(DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, BOTTOM))
bcs.append(DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, CYL))
# Pressure reference at outlet (traction-free outflow with p=0)
bcs.append(DirichletBC(W.sub(1), Constant(0.0), boundaries, RIGHT))

# -----------------------
# Initial condition
# -----------------------
# u = (0,0), p = 0 with a tiny perturbation to encourage shedding (optional)
assign(w_n.sub(0), project(Constant((0.0, 0.0)), V))
assign(w_n.sub(1), project(Constant(0.0), Q))
# Tiny broadband perturbation on velocity to break symmetry (you may comment out if not desired)
rng = np.random.RandomState(1)
u_vec = w_n.sub(0).vector().get_local()
u_vec += 1e-4 * (rng.randn(u_vec.size))
w_n.sub(0).vector().set_local(u_vec)
w_n.sub(0).vector().apply("insert")

# -----------------------
# Helpers: time discretisation & stabilisation
# -----------------------
k  = Constant(dt)
nu_c = Constant(nu)
rho_c = Constant(rho)
U_char = Constant(U_in)
D_char = Constant(D)

# Semi-implicit convection with Picard linearisation using u_conv = u_n
u_conv = u_n

def symgrad(w):  return sym(grad(w))

# Element size and velocity magnitude for stabilisation
# Use cell-wise metric via h = cell diameter
h = CellDiameter(mesh)
umag = sqrt(inner(u_conv, u_conv) + DOLFIN_EPS)

# Residuals (based on semi-implicit form)
# Momentum residual R_m ~ (u - u_n)/dt + (u_conv · ∇)u + ∇p - ν Δu
# (Use grad-grad form for viscous term.)
R_m = ( (u - u_n)/k + grad(u)*u_conv + grad(p) - nu_c*div(grad(u)) )

# Continuity residual
R_c = div(u)

# Stabilisation parameters (classic choices)
# tau_m (a.k.a. tau_SUPG/PSPG) and tau_c for continuity
# Here use a simple robust form:
tau_m = 1.0 / sqrt( (2.0/k)**2 + (C_tau*umag/h)**2 + (C_tau*nu_c/(h*h))**2 )
tau_c = 1.0 / ( (2.0/k) + C_tau*umag/h )

# -----------------------
# Weak form (θ-scheme with θ=1: backward Euler)
# -----------------------
# Standard Galerkin terms
a_gal = rho_c*inner((u - u_n)/k, v)*dx \
      + rho_c*inner(grad(u)*u_conv, v)*dx \
      + 2.0*nu_c*inner(symgrad(u), symgrad(v))*dx \
      - div(v)*p*dx + q*div(u)*dx

# Grad–div stabilisation
a_gd = gamma_gd * inner(div(u), div(v)) * dx

# SUPG + PSPG (RBVMS residual-based, linearised as above)
a_supg = supg_scale * tau_m * rho_c * inner(R_m, grad(v)*u_conv) * dx
a_pspg = pspg_scale * tau_m * inner(R_m, grad(q)) * dx

# Optional continuity stabilisation (least-squares on mass residual)
a_cont = tau_c * inner(R_c, div(v)) * dx

F = a_gal + a_gd + a_supg + a_pspg + a_cont

# Convert to linear system (Picard each step). We’ll solve once per step with u_conv=u_n.
A = None
solver = LUSolver()   # change to Krylov for larger runs
solver.parameters["reuse_factorization"] = True

# -----------------------
# Post-processing: drag on cylinder and mean Cd over [8,10] s
# -----------------------
mu = Constant(rho*nu)  # dynamic viscosity
I  = Identity(mesh.geometry().dim())
sigma = -p_*I + 2.0*mu*symgrad(u_)

drag_dir = Constant((1.0, 0.0))  # x-direction
Fx_form  = -inner(dot(sigma, n), drag_dir)*ds_(CYL)  # force on fluid boundary ⇒ minus traction on cylinder
# Note: sign chosen so that positive Fx_form is drag resisting the flow

denom_cd = 0.5*rho*(U_in**2)*D

t = 0.0
drag_mean_num = 0.0
drag_mean_den = 0.0

# Files
xfile = XDMFFile(mesh.mpi_comm(), "vms_solution.xdmf")
xfile.parameters["flush_output"] = True
xfile.parameters["functions_share_mesh"] = True
# (we will write only final fields as requested)

# Time-stepping
num_steps = int(round(t_end/dt))
for step in range(1, num_steps+1):
    t = step*dt

    # Assemble & solve the linearised step
    if A is None:
        A = assemble(lhs(F))
    else:
        # Re-assemble if you tweak parameters; with fixed form, reuse A
        pass
    b = assemble(rhs(F))
    for bc in bcs:
        bc.apply(A, b)

    w_.vector()[:] = 0.0
    solver.solve(A, w_.vector(), b)

    # Update for next step (Picard with one sweep)
    w_n.assign(w_)

    # Drag and Cd
    Fx = assemble(Fx_form)
    Cd = Fx / denom_cd

    # Accumulate mean over [8,10] s via trapezoidal rectangle (uniform dt)
    if t >= 8.0 - 1e-12:
        drag_mean_num += Cd*dt
        drag_mean_den += dt

    # Optional: simple progress print every 100 steps
    if step % 100 == 0 or abs(t - t_end) < 1e-12:
        print("t = %.3f  |  Fx = %.6e  |  Cd = %.6f" % (t, Fx, Cd))

# Report mean Cd
Cd_mean = drag_mean_num/drag_mean_den if drag_mean_den > 0.0 else float("nan")
print("\nMean drag coefficient over [8.0, 10.0] s:  %.6f" % Cd_mean)

# Save final fields at t=10 s
(u_final, p_final) = w_.split(deepcopy=True)
u_final.rename("u", "velocity")
p_final.rename("p", "pressure")
xfile.write(u_final, t)
xfile.write(p_final, t)
xfile.close()