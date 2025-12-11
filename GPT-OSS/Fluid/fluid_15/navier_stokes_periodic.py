# filename: navier_stokes_periodic.py
from __future__ import print_function
from fenics import *
import numpy as np

# ----------------------------------------------------------------------
# 1. Periodic boundary definition
# ----------------------------------------------------------------------
class PeriodicBoundary(SubDomain):
    """Map the right edge to the left edge and the top edge to the bottom edge."""
    def inside(self, x, on_boundary):
        # left or bottom edge (excluding the corner (0,0))
        return bool(((near(x[0], 0) or near(x[1], 0)) and
                     (not (near(x[0], 0) and near(x[1], 0)))) and on_boundary)

    def map(self, x, y):
        if near(x[0], 1) and near(x[1], 1):
            # corner (1,1) → (0,0)
            y[0] = x[0] - 1.0
            y[1] = x[1] - 1.0
        elif near(x[0], 1):
            # right edge → left edge
            y[0] = x[0] - 1.0
            y[1] = x[1]
        else:   # near(x[1],1)
            # top edge → bottom edge
            y[0] = x[0]
            y[1] = x[1] - 1.0

# ----------------------------------------------------------------------
# 2. Mesh and function spaces (Taylor‑Hood, periodic)
# ----------------------------------------------------------------------
N = 32                                 # mesh resolution (feel free to increase)
mesh = UnitSquareMesh(N, N, "crossed")
pbc = PeriodicBoundary()

# Define the finite elements (without constrained_domain)
V_el = VectorElement("CG", mesh.ufl_cell(), 2)   # P2 velocity
Q_el = FiniteElement("CG", mesh.ufl_cell(), 1)   # P1 pressure

# Build the mixed element and the mixed function space, attaching the periodicity
W_el = MixedElement([V_el, Q_el])
W = FunctionSpace(mesh, W_el, constrained_domain=pbc)

# Subspaces (use the same periodic constraint)
V = FunctionSpace(mesh, V_el, constrained_domain=pbc)
Q = FunctionSpace(mesh, Q_el, constrained_domain=pbc)

# ----------------------------------------------------------------------
# 3. Trial / test functions and solution containers
# ----------------------------------------------------------------------
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

w   = Function(W)          # (u^{n+1}, p^{n+1})
w0  = Function(W)          # (u^{n},   p^{n})

# Split the mixed functions for convenient handling
(u0, p0) = split(w0)       # previous velocity & pressure (UFL expressions)

# ----------------------------------------------------------------------
# 4. Physical parameters
# ----------------------------------------------------------------------
rho = Constant(1.0)               # density
nu  = Constant(1e-3)              # kinematic viscosity

# ----------------------------------------------------------------------
# 5. Initial condition (analytical field)
# ----------------------------------------------------------------------
u0_expr = Expression(("sin(2*pi*x[0])*cos(2*pi*x[1])",
                     "-cos(2*pi*x[0])*sin(2*pi*x[1])"),
                     degree=5, pi=np.pi)

# Interpolate the initial velocity and pressure into the subspaces
u0_func = interpolate(u0_expr, V)
p0_func = interpolate(Constant(0.0), Q)

# Populate the mixed function w0 with the initial fields
assign(w0.sub(0), u0_func)
assign(w0.sub(1), p0_func)

# ----------------------------------------------------------------------
# 6. Time stepping parameters
# ----------------------------------------------------------------------
T  = 1.0
dt = 0.005                     # small enough for CFL ≈ 0.1 on this mesh
num_steps = int(T / dt)

# Times at which we write output
output_times = [0.0, 0.25, 0.5, 1.0]

# XDMF writer (velocity only, as requested)
xdmf_file = XDMFFile(mesh.mpi_comm(), "velocity_periodic.xdmf")
xdmf_file.parameters["flush_output"] = True
xdmf_file.parameters["functions_share_mesh"] = True

# Write the initial velocity
xdmf_file.write(u0_func, 0.0)

# ----------------------------------------------------------------------
# 7. Variational formulation (semi‑implicit backward Euler)
# ----------------------------------------------------------------------
U = u0_func                     # convection velocity from previous step (explicit)

def epsilon(u):
    return sym(grad(u))

def sigma(u, p):
    return 2*nu*epsilon(u) - p*Identity(len(u))

F = (rho*dot((u - u0)/dt, v) * dx
     + rho*dot(dot(U, nabla_grad(u)), v) * dx
     + inner(sigma(u, p), epsilon(v)) * dx
     - div(v)*p * dx
     - q*div(u) * dx)

a, L = lhs(F), rhs(F)

# ----------------------------------------------------------------------
# 8. Time loop
# ----------------------------------------------------------------------
t = 0.0
for n in range(1, num_steps + 1):
    t += dt

    # Solve the linear system (convection treated explicitly → linear)
    solve(a == L, w, solver_parameters={"linear_solver": "mumps"})

    # Extract the components and store them for the next step
    (u_sol, p_sol) = w.split()
    assign(w0.sub(0), u_sol)
    assign(w0.sub(1), p_sol)

    # Output at the requested times (tiny tolerance to avoid floating‑point issues)
    if any(abs(t - tout) < 1e-8 for tout in output_times):
        print("Saving velocity at t = {:.3f}".format(t))
        xdmf_file.write(u_sol, t)

# ----------------------------------------------------------------------
# 9. Clean up
# ----------------------------------------------------------------------
xdmf_file.close()
print("Simulation finished.")