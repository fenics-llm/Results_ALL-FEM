# -*- coding: utf-8 -*-
#
# 2D Fluid–Structure Interaction in an elastic tube (legacy FEniCS)
#
#  Geometry: fluid 6 cm × 1 cm, upper/lower walls 0.1 cm thick.
#  Time step: Δt = 1e‑4 s, output at t = 0.005 s and t = 0.100 s.
#
#  NOTE: This script uses the legacy DOLFIN (fenics) API only.
#
from fenics import *
import numpy as np

# ----------------------------------------------------------------------
# 1. Mesh and sub‑domain markers
# ----------------------------------------------------------------------
L, H = 6.0, 1.0          # fluid size (cm)
t_wall = 0.1             # wall thickness (cm)

mesh = RectangleMesh(Point(0.0, -t_wall), Point(L, H + t_wall), 120, 80)

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)

class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and on_boundary
class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], L) and on_boundary
class UpperOuter(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], H + t_wall) and on_boundary
class LowerOuter(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], -t_wall) and on_boundary
class InterfaceTop(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], H) and on_boundary
class InterfaceBot(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0) and on_boundary
class SideWalls(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[0], 0.0) or near(x[0], L)) and on_boundary
class PressurePoint(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and near(x[1], 0.0) and on_boundary

Inlet().mark(boundaries, 1)
Outlet().mark(boundaries, 2)
UpperOuter().mark(boundaries, 3)
LowerOuter().mark(boundaries, 4)
InterfaceTop().mark(boundaries, 5)
InterfaceBot().mark(boundaries, 6)

ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# ----------------------------------------------------------------------
# 2. Material parameters (CGS)
# ----------------------------------------------------------------------
mu_f   = 0.003          # fluid viscosity (g·cm⁻¹·s⁻¹)
rho_f  = 1.0            # fluid density (g·cm⁻³)

rho_s  = 1.1            # solid density (g·cm⁻³)
nu_s   = 0.49
E_s    = 3.0e5          # solid Young modulus (Pa = g·cm⁻¹·s⁻²)

lambda_s = E_s*nu_s/((1 + nu_s)*(1 - 2*nu_s))
mu_s     = E_s/(2*(1 + nu_s))

# ----------------------------------------------------------------------
# 3. Function spaces (Taylor–Hood for fluid, mixed u‑p for solid)
# ----------------------------------------------------------------------
V_f = VectorFunctionSpace(mesh, "Lagrange", 2)
Q_f = FunctionSpace(mesh, "Lagrange", 1)
W_f = FunctionSpace(mesh, MixedElement([V_f.ufl_element(),
                                    Q_f.ufl_element()]))

V_s = VectorFunctionSpace(mesh, "Lagrange", 2)
Q_s = FunctionSpace(mesh, "Lagrange", 1)
W_s = FunctionSpace(mesh, MixedElement([V_s.ufl_element(),
                                        Q_s.ufl_element()]))

V_mesh = VectorFunctionSpace(mesh, "Lagrange", 1)

# ----------------------------------------------------------------------
# 4. Trial / test functions
# ----------------------------------------------------------------------
(v_f, p_f) = TrialFunctions(W_f)
(v_f_test, q_f_test) = TestFunctions(W_f)

(d_s, p_s) = TrialFunctions(W_s)
(v_s_test, q_s_test) = TestFunctions(W_s)

# ----------------------------------------------------------------------
# 5. Previous step functions
# ----------------------------------------------------------------------
w_f_n = Function(W_f)          # fluid (v_f, p_f) at previous step
w_s_n = Function(W_s)          # solid (d_s, p_s) at previous step
w_mesh_n = Function(V_mesh)    # mesh displacement at previous step

(v_f_n, p_f_n) = w_f_n.split(deepcopy=True)
(d_s_n, p_s_n) = w_s_n.split(deepcopy=True)

# ----------------------------------------------------------------------
# 5b. Current step functions
# ----------------------------------------------------------------------
w_f = Function(W_f)          # fluid solution at new step
w_s = Function(W_s)          # solid solution at new step

# ----------------------------------------------------------------------
# 6. Mesh velocity (ALE)
# ----------------------------------------------------------------------
dt = Constant(1e-4)   # time step
w_mesh = Function(V_mesh)               # current mesh displacement
w_mesh_vel = Function(V_mesh)           # mesh velocity

# ----------------------------------------------------------------------
# 7. Constitutive relations
# ----------------------------------------------------------------------
def sigma_f(v, p):
    return -p*Identity(2) + 2*mu_f*sym(grad(v))

def sigma_s(d, p):
    eps = sym(grad(d))
    return -p*Identity(2) + 2*mu_s*eps + lambda_s*tr(eps)*Identity(2)

# ----------------------------------------------------------------------
# 8. Boundary conditions
# ----------------------------------------------------------------------
bc_fluid_top = DirichletBC(W_f.sub(0), w_mesh_vel, InterfaceTop())
bc_fluid_bot = DirichletBC(W_f.sub(0), w_mesh_vel, InterfaceBot())

p0 = PressurePoint()
bc_pressure = DirichletBC(W_f.sub(1), Constant(0.0), p0, "pointwise")
bc_fluid = [bc_fluid_top, bc_fluid_bot, bc_pressure]

side_walls = SideWalls()
bc_solid_side = DirichletBC(W_s.sub(0).sub(0), Constant(0.0), side_walls, "pointwise")
bc_solid = [bc_solid_side]

# ----------------------------------------------------------------------
# 9. Time stepping
# ----------------------------------------------------------------------
t = 0.0
t_end = 0.1
output_times = [0.005, 0.100]

xdmf = XDMFFile(mesh.mpi_comm(), "fsi_results.xdmf")
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True

while t < t_end + 1e-12:
    # ------------------------------------------------------------------
    # 9.1 Mesh update (Laplace smoothing) – Dirichlet = solid displacement (vector)
    # ------------------------------------------------------------------
    bc_mesh_top = DirichletBC(V_mesh, d_s_n, InterfaceTop())
    bc_mesh_bot = DirichletBC(V_mesh, d_s_n, InterfaceBot())
    d_mesh = TrialFunction(V_mesh)
    v_mesh = TestFunction(V_mesh)
    a_mesh = inner(grad(d_mesh), grad(v_mesh))*dx
    L_mesh = inner(Constant((0.0, 0.0)), v_mesh)*dx
    solve(a_mesh == L_mesh, w_mesh, [bc_mesh_top, bc_mesh_bot])

    # mesh velocity for this step
    w_mesh_vel.vector()[:] = (w_mesh.vector() - w_mesh_n.vector())/float(dt)

    # ------------------------------------------------------------------
    # 9.2 Fluid problem (backward Euler, ALE) – convective term omitted
    # ------------------------------------------------------------------
    F_fluid = (rho_f/dt)*inner(v_f, v_f_test)*dx \
               + 2*mu_f*inner(sym(grad(v_f)), sym(grad(v_f_test)))*dx \
               - p_f*div(v_f_test)*dx \
               - q_f_test*div(v_f)*dx \
               - (rho_f/dt)*inner(v_f_n, v_f_test)*dx

    if t < 5e-3:
        T_in = - (2e4/2.0)*(1.0 - np.cos(np.pi*t/2.5e-3))
    else:
        T_in = 0.0
    F_fluid += T_in*dot(v_f_test, FacetNormal(mesh))*ds(1)

    solve(lhs(F_fluid) == rhs(F_fluid), w_f, bc_fluid)

    # ------------------------------------------------------------------
    # 9.3 Solid problem (static linear elasticity, mixed u‑p)
    # ------------------------------------------------------------------
    (v_f_sol, p_f_sol) = w_f.split(deepcopy=True)
    traction = dot(sigma_f(v_f_sol, p_f_sol), FacetNormal(mesh))
    F_solid = mu_s*inner(grad(d_s), grad(v_s_test))*dx \
               - p_s*div(v_s_test)*dx \
               - q_s_test*(lambda_s*div(d_s) + p_s)*dx \
               - dot(traction, v_s_test)*(ds(5) + ds(6))

    solve(lhs(F_solid) == rhs(F_solid), w_s, bc_solid)

    # ------------------------------------------------------------------
    # 9.4 Update previous step fields
    # ------------------------------------------------------------------
    w_f_n.assign(w_f)
    w_s_n.assign(w_s)
    w_mesh_n.assign(w_mesh)

    # ------------------------------------------------------------------
    # 9.5 Output at requested times
    # ------------------------------------------------------------------
    if any(abs(t - tout) < 1e-12 for tout in output_times):
        (v_f_sol, p_f_sol) = w_f.split(deepcopy=True)
        xdmf.write(v_f_sol, t)   # fluid velocity
        (d_s_sol, p_s_sol) = w_s.split(deepcopy=True)
        xdmf.write(d_s_sol, t)   # wall displacement

    # ------------------------------------------------------------------
    # 9.6 Advance time
    # ------------------------------------------------------------------
    t += float(dt)

xdmf.close()