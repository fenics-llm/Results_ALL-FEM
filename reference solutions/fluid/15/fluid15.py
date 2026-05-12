# filename: navier_stokes_periodic.py
from __future__ import print_function
from dolfin import *
import numpy as np

# ----------------------------------------------------------------------
# 1. Periodic boundary definition
# ----------------------------------------------------------------------
class PeriodicBoundary(SubDomain):
    """
    Periodic on unit square:
      x = 1 maps to x = 0
      y = 1 maps to y = 0
      corner (1,1) maps to (0,0)
    """

    def inside(self, x, on_boundary):
        return bool(
            on_boundary and
            (
                near(x[0], 0.0) or near(x[1], 0.0)
            ) and
            not (
                near(x[0], 1.0) or near(x[1], 1.0)
            )
        )

    def map(self, x, y):
        if near(x[0], 1.0) and near(x[1], 1.0):
            y[0] = x[0] - 1.0
            y[1] = x[1] - 1.0
        elif near(x[0], 1.0):
            y[0] = x[0] - 1.0
            y[1] = x[1]
        elif near(x[1], 1.0):
            y[0] = x[0]
            y[1] = x[1] - 1.0
        else:
            y[0] = x[0]
            y[1] = x[1]

# ----------------------------------------------------------------------
# 2. Mesh and periodic function space
# ----------------------------------------------------------------------
N = 32
mesh = UnitSquareMesh(N, N, "crossed")
pbc = PeriodicBoundary()

V_el = VectorElement("CG", mesh.ufl_cell(), 2)
Q_el = FiniteElement("CG", mesh.ufl_cell(), 1)
W_el = MixedElement([V_el, Q_el])

W = FunctionSpace(mesh, W_el, constrained_domain=pbc)
V = FunctionSpace(mesh, V_el, constrained_domain=pbc)
Q = FunctionSpace(mesh, Q_el, constrained_domain=pbc)

# ----------------------------------------------------------------------
# 3. Unknowns and tests
# ----------------------------------------------------------------------
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

w = Function(W)
w0 = Function(W)

u0, p0 = split(w0)

# ----------------------------------------------------------------------
# 4. Parameters
# ----------------------------------------------------------------------
rho = Constant(1.0)
nu = Constant(1.0e-3)

# ----------------------------------------------------------------------
# 5. Initial condition
# ----------------------------------------------------------------------
u0_expr = Expression(
    (
        "sin(2*pi*x[0])*cos(2*pi*x[1])",
        "-cos(2*pi*x[0])*sin(2*pi*x[1])"
    ),
    degree=5,
    pi=np.pi
)

u_init = interpolate(u0_expr, V)
p_init = interpolate(Constant(0.0), Q)

assign(w0.sub(0), u_init)
assign(w0.sub(1), p_init)

# ----------------------------------------------------------------------
# 6. Time stepping
# ----------------------------------------------------------------------
T = 1.0
dt = 0.0025
num_steps = int(round(T/dt))

output_times = [0.0, 0.25, 0.5, 1.0]
output_tol = 0.5*dt + 1.0e-12

# ----------------------------------------------------------------------
# 7. Pressure gauge
# ----------------------------------------------------------------------
class PressureGauge(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and near(x[1], 0.0)

bc_p = DirichletBC(W.sub(1), Constant(0.0), PressureGauge(), method="pointwise")
bcs = [bc_p]

# ----------------------------------------------------------------------
# 8. Weak form: semi-implicit backward Euler
#
# Unknown u = u^{n+1}
# Previous velocity u0 = u^n
# Convection linearized as (u0 · grad) u
# ----------------------------------------------------------------------
def epsilon(a):
    return sym(grad(a))

F = (
    rho*dot((u - u0)/dt, v)*dx
    + rho*dot(dot(u0, nabla_grad(u)), v)*dx
    + 2.0*nu*inner(epsilon(u), epsilon(v))*dx
    - p*div(v)*dx
    + q*div(u)*dx
)

a = lhs(F)
Lform = rhs(F)

# ----------------------------------------------------------------------
# 9. Output file
# ----------------------------------------------------------------------
u_init.rename("u", "velocity")

xdmf = XDMFFile(mesh.mpi_comm(), "velocity_periodic.xdmf")
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True
xdmf.write(u_init, 0.0)

# ----------------------------------------------------------------------
# 10. Time loop
# ----------------------------------------------------------------------
t = 0.0

for step in range(1, num_steps + 1):
    t = step*dt

    solve(
        a == Lform,
        w,
        bcs,
        solver_parameters={"linear_solver": "mumps"}
    )

    u_sol, p_sol = w.split(deepcopy=True)

    assign(w0.sub(0), u_sol)
    assign(w0.sub(1), p_sol)

    if any(abs(t - tout) <= output_tol for tout in output_times):
        print("Saving velocity at t = %.4f" % t)
        u_sol.rename("u", "velocity")
        xdmf.write(u_sol, t)

xdmf.close()
print("Simulation finished.")