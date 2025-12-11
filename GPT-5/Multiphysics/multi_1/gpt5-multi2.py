# Legacy FEniCS: tested with dolfin 2019.1.0 and mshr
from dolfin import *
from mshr import Circle, generate_mesh
import math

# -------------------------
# Parameters
# -------------------------
R0 = 0.05                       # m
s  = 1.0e-4                     # m/s (radial expansion speed)
D  = 1.0e-5                     # m^2/s
kappa = 1.0e-4                  # 1/s
dt = 0.01                       # s
Tfinal = 10.0                   # s
nsteps = int(round(Tfinal/dt))  # 1000

# -------------------------
# Initial mesh (unstructured disk) with h0 ~ 1e-3
# -------------------------
domain = Circle(Point(0.0, 0.0), R0)
mesh = generate_mesh(domain, 100)  # start with a reasonable resolution

# Refine until coarse size is about 1e-3 (tolerances are loose on purpose)
while mesh.hmax() > 1.2e-3:
    mesh = refine(mesh)

# -------------------------
# Function spaces
# -------------------------
V  = FunctionSpace(mesh, "CG", 1)
Vv = VectorFunctionSpace(mesh, "CG", 1)

# Mesh-velocity (radially outward, |w| = s), safe at the origin
class MeshVelocity(UserExpression):
    def __init__(self, s, **kwargs):
        self.s = s
        super().__init__(**kwargs)
    def eval(self, value, x):
        r = math.hypot(x[0], x[1])
        if r < 1.0e-14:
            value[0] = 0.0
            value[1] = 0.0
        else:
            value[0] = self.s * x[0]/r
            value[1] = self.s * x[1]/r
    def value_shape(self):
        return (2,)

# Time-dependent fields
c_n = Function(V)            # previous time level
c   = Function(V)            # current solution
w_expr = MeshVelocity(s, degree=1)
w = interpolate(w_expr, Vv)  # mesh velocity field on the current mesh

# Initial condition: c(x,0) = 1 in Omega(0)
c_n.assign(Constant(1.0))

# -------------------------
# Variational formulation (Backward Euler in time)
# (dc/dt, v) + (D∇c, ∇v) + ((w c)·∇v) = -(kappa c, v)
# -------------------------
u  = TrialFunction(V)
v  = TestFunction(V)

a_form = ( (1.0/dt)*u*v*dx
           + D*dot(grad(u), grad(v))*dx
           + dot(w, grad(v))*u*dx
           + kappa*u*v*dx )

L_form = ( (1.0/dt)*c_n*v*dx )

A = None  # we'll assemble each step after updating w (since mesh moves)

# -------------------------
# XDMF output
# -------------------------
c.rename("c", "concentration")
xfile = XDMFFile(MPI.comm_world, "concentration_timeseries.xdmf")
xfile.parameters["flush_output"] = True
xfile.parameters["functions_share_mesh"] = True
xfile.parameters["rewrite_function_mesh"] = False  # moving mesh is written each step anyway

# Write initial state (t=0) on initial mesh
xfile.write(c_n, 0.0)

# Helper to compute total concentration
def total_concentration(f):
    return assemble(f*dx)

# Report initial total concentration
if MPI.comm_world.rank == 0:
    M0 = total_concentration(c_n)
    print(f"Step 0, t = {0.0:.2f} s : total concentration = {M0:.8e}")

# -------------------------
# Time-stepping
# -------------------------
t = 0.0
for n in range(1, nsteps+1):
    # Target time
    t = n*dt

    # --- Move mesh by ALE: x^{n+1} = x^{n} + dt * w(x^n)
    # Recompute w on the *current* mesh (depends on coordinates)
    w.assign(interpolate(w_expr, Vv))
    disp = project(w*dt, Vv)          # displacement over this step
    ALE.move(mesh, disp)

    # After mesh move, update function spaces' mappings implicitly (same mesh object)
    # Re-interpolate w on the moved mesh to be consistent in the forms (w^{n+1})
    w.assign(interpolate(w_expr, Vv))

    # Assemble system on the moved mesh
    a = assemble(a_form)
    L = assemble(L_form)

    # Solve
    solve(a, c.vector(), L, "cg", "ilu")

    # Output
    xfile.write(c, t)

    # Every 100 steps, report the total concentration
    if n % 100 == 0:
        M = total_concentration(c)
        if MPI.comm_world.rank == 0:
            print(f"Step {n:4d}, t = {t:5.2f} s : total concentration = {M:.8e}")

    # Update for next step
    c_n.assign(c)

# Final report
if MPI.comm_world.rank == 0:
    M = total_concentration(c)
    print(f"Final, t = {t:.2f} s : total concentration = {M:.8e}")