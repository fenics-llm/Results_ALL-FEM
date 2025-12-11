# -*- coding: utf-8 -*-
#
# Turbulent flow over a circular cylinder (VMS‑SUPG‑grad‑div) – legacy FEniCS
#
# Domain:  Ω = [-30,30]×[-30,30]  (D = 1.0 m)
# Cylinder: centre (0,0), radius R = D/2 = 0.5
#
# Physical parameters
U  = 1.0          # inflow velocity (m/s)
nu = 2.56e-5      # kinematic viscosity (m^2/s)
rho = 1.0         # density (kg/m^3)
mu = rho*nu       # dynamic viscosity (Pa·s)
D  = 1.0
R  = D/2.0

# Time stepping
T   = 10.0        # final time (s)
dt  = 0.001       # time step (s)
num_steps = int(T/dt)

# --------------------------------------------------------------
# 1. Mesh with a circular hole
# --------------------------------------------------------------
from dolfin import *
import mshr
from math import pi          # needed for the perturbation expression

# Build a rectangle and subtract a circle
rect = mshr.Rectangle(Point(-30.0, -30.0), Point(30.0, 30.0))
cyl  = mshr.Circle(Point(0.0, 0.0), R, 64)   # 64 points on the circle
mesh = mshr.generate_mesh(rect - cyl, 80)      # global mesh size ≈ 80

# --------------------------------------------------------------
# 2. Boundary definitions
# --------------------------------------------------------------
tol = 1E-10

class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], -30.0, tol)

class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 30.0, tol)

class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], -30.0, tol) or near(x[1], 30.0, tol))

class Cylinder(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (x[0]**2 + x[1]**2 < (R+tol)**2)

inlet   = Inlet()
outlet  = Outlet()
walls   = Walls()
cylinder = Cylinder()

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
inlet.mark(boundaries, 1)
outlet.mark(boundaries, 2)
walls.mark(boundaries, 3)
cylinder.mark(boundaries, 4)

ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# --------------------------------------------------------------
# 3. Function spaces (Taylor–Hood)
# --------------------------------------------------------------
Ve = VectorElement("Lagrange", mesh.ufl_cell(), 2)   # velocity (P2)
Pe = FiniteElement("Lagrange", mesh.ufl_cell(), 1)   # pressure (P1)
TH = MixedElement([Ve, Pe])
W  = FunctionSpace(mesh, TH)

# --------------------------------------------------------------
# 4. Trial / test functions and solution containers
# --------------------------------------------------------------
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

w   = Function(W)          # (u^{n+1}, p^{n+1})
w0  = Function(W)          # (u^{n},   p^{n})
(u0, p0) = split(w0)

# --------------------------------------------------------------
# 5. Boundary conditions
# --------------------------------------------------------------
# Inlet velocity profile (uniform)
inlet_profile = Expression(("U", "0.0"), U=U, degree=2)

bcu_inlet   = DirichletBC(W.sub(0), inlet_profile, inlet)
bcu_walls   = DirichletBC(W.sub(0), Constant((0.0, 0.0)), walls)
bcu_cyl     = DirichletBC(W.sub(0), Constant((0.0, 0.0)), cylinder)

# Pressure gauge at a single point on the outlet (to fix nullspace)
class PressurePoint(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 30.0, tol) and near(x[1], 0.0, tol)

p_point = PressurePoint()
bc_p = DirichletBC(W.sub(1), Constant(0.0), p_point, method="pointwise")

bcs = [bcu_inlet, bcu_walls, bcu_cyl, bc_p]

# --------------------------------------------------------------
# 6. VMS‑SUPG‑grad‑div parameters (elementwise)
# --------------------------------------------------------------
h = CellDiameter(mesh)
u_norm = sqrt(dot(u0, u0) + DOLFIN_EPS)   # |u^n|
C_I = 4.0
C_G = 0.1

# Corrected τ_M (2/(ρΔt) rather than (2/ρ)/Δt)
tau_M = 1.0 / sqrt( (2.0/(rho*dt))**2
                     + (u_norm*C_I/h)**2
                     + (C_I*mu/(rho*h**2))**2 )
tau_C = tau_M * (rho*C_I**2)
tau_G = C_G * h**2

# --------------------------------------------------------------
# 7. Residuals
# --------------------------------------------------------------
R_M = rho*((u - u0)/dt + dot(u0, nabla_grad(u))) \
      + grad(p) - 2*mu*div(sym(grad(u)))

R_C = div(u)

# --------------------------------------------------------------
# 8. Weak form (VMS‑SUPG‑grad‑div)
# --------------------------------------------------------------
F = ( rho*dot((u - u0)/dt, v)*dx
      + rho*dot(dot(u0, nabla_grad(u)), v)*dx
      + 2*mu*inner(sym(grad(u)), sym(grad(v)))*dx
      - div(v)*p*dx
      + q*div(u)*dx
      + tau_M*dot(R_M, rho*dot(u0, nabla_grad(v)) + 2*mu*div(sym(grad(v))) - nabla_grad(q))*dx
      + tau_C*R_C*div(v)*dx
      + tau_G*rho*div(u)*rho*div(v)*dx )

a = lhs(F)
L = rhs(F)

# --------------------------------------------------------------
# 9. Drag coefficient functional
# --------------------------------------------------------------
n = FacetNormal(mesh)
stress = -p*Identity(2) + 2*mu*sym(grad(u))
drag_form = dot(stress*n, Constant((1.0, 0.0)))*ds(4)   # cylinder marker = 4

# --------------------------------------------------------------
# 10. Time loop
# --------------------------------------------------------------
drag_vals = []
t = 0.0

# XDMF output
xdmf = XDMFFile(mesh.mpi_comm(), "vms_solution.xdmf")
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True

# Initial condition (zero + tiny perturbation to trigger shedding)
w0.vector()[:] = 0.0
# -----------------------------------------------------------------
#   FIX: use x[1] (the y‑coordinate) instead of undefined variable y
# -----------------------------------------------------------------
pert = Expression(("eps*sin(2*pi*x[1]/D)", "0.0"),
                  eps=1e-4, D=D, pi=pi, degree=2)
assign(w0.sub(0), interpolate(pert, W.sub(0).collapse()))

for step in range(num_steps):
    t += dt

    # Assemble and solve
    A = assemble(a)
    b = assemble(L)
    [bc.apply(A, b) for bc in bcs]
    solve(A, w.vector(), b, "mumps")

    # Split solution
    (u_sol, p_sol) = w.split(deepcopy=True)

    # Drag at this time step
    drag = assemble(drag_form)
    drag_vals.append(drag)

    # Write to XDMF (velocity and pressure)
    xdmf.write(u_sol, t)
    xdmf.write(p_sol, t)

    # Update previous solution
    w0.assign(w)

# --------------------------------------------------------------
# 11. Time‑averaged drag over [8,10] s
# --------------------------------------------------------------
import numpy as np
times = np.arange(dt, T+dt, dt)
mask = (times >= 8.0) & (times <= 10.0)
Cd = np.array(drag_vals) * 2.0 / (rho*U**2*D)   # drag coefficient
Cd_mean = np.mean(Cd[mask])

print("\nMean drag coefficient over [8,10] s : {:.6f}".format(Cd_mean))

xdmf.close()