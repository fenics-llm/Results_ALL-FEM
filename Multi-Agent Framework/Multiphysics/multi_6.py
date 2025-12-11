# -*- coding: utf-8 -*-
#
#  2‑D fluid–structure interaction: flag in a channel (legacy FEniCS)
#
#  Geometry: channel 2.5 × 0.41 m, rigid pole (disk r=0.05 m, centre (0.20,0.20) m)
#            flag rectangle 0.35 × 0.02 m, bottom‑right corner at (0.60,0.20) m
#
#  Fluid: ρ_f = 1000 kg/m³, ν_f = 1e-3 m²/s  → μ_f = ρ_f·ν_f = 1.0 Pa·s
#  Solid: ρ_s = 10000 kg/m³, ν_s = 0.4, μ_s = 0.5e6 Pa,
#          λ_s = 2·μ_s·ν_s/(1‑2·ν_s)
#
#  Time integration: Δt = 0.001 s, T = 4.0 s
#
#  Output: velocity, pressure (fluid) and displacement (solid) in XDMF,
#          point‑A displacement (A₀ = (0.60,0.20) m) in a CSV file.
#
# -----------------------------------------------------------------
# 1. Geometry & mesh
# -----------------------------------------------------------------
from dolfin import *
from mshr import Rectangle, Circle, generate_mesh
from math import pi, cos

L, H = 2.5, 0.41
pole_center = Point(0.20, 0.20)
pole_radius = 0.05
flag_bl = Point(0.60, 0.19)               # bottom‑left corner
flag_tr = Point(0.60 + 0.35, 0.19 + 0.02)  # top‑right corner

channel = Rectangle(Point(0.0, 0.0), Point(L, H))
pole    = Circle(pole_center, pole_radius, 64)
flag    = Rectangle(flag_bl, flag_tr)

fluid_ref = channel - pole
solid_ref = flag - pole
mesh = generate_mesh(fluid_ref + solid_ref, 64)   # global resolution

# -----------------------------------------------------------------
# 2. Sub‑domains & boundaries
# -----------------------------------------------------------------
class FluidDomain(SubDomain):
    def inside(self, x, on_boundary):
        return (0.0 <= x[0] <= L) and (0.0 <= x[1] <= H) and \
               ((x[0]-pole_center[0])**2 + (x[1]-pole_center[1])**2 >= pole_radius**2 - DOLFIN_EPS)

class SolidDomain(SubDomain):
    def inside(self, x, on_boundary):
        return (flag_bl[0] <= x[0] <= flag_tr[0]) and \
               (flag_bl[1] <= x[1] <= flag_tr[1]) and \
               ((x[0]-pole_center[0])**2 + (x[1]-pole_center[1])**2 >= pole_radius**2 - DOLFIN_EPS)

class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and on_boundary

class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], L) and on_boundary

class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[1], 0.0) or near(x[1], H)) and on_boundary

class PoleBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return ((x[0]-pole_center[0])**2 + (x[1]-pole_center[1])**2 <= pole_radius**2 + DOLFIN_EPS) and on_boundary

subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
subdomains.set_all(0)
FluidDomain().mark(subdomains, 1)
SolidDomain().mark(subdomains, 2)

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
Inlet().mark(boundaries, 1)
Outlet().mark(boundaries, 2)
Walls().mark(boundaries, 3)
PoleBoundary().mark(boundaries, 4)

dx = Measure('dx', domain=mesh, subdomain_data=subdomains)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# -----------------------------------------------------------------
# 3. Function spaces (Taylor–Hood + mesh displacement)
# -----------------------------------------------------------------
V_f = VectorFunctionSpace(mesh, "Lagrange", 2)   # fluid velocity
Q_f = FunctionSpace(mesh, "Lagrange", 1)         # fluid pressure
V_s = VectorFunctionSpace(mesh, "Lagrange", 2)   # solid displacement
V_m = VectorFunctionSpace(mesh, "Lagrange", 2)   # mesh displacement

mixed_elem = MixedElement([V_f.ufl_element(),
                           Q_f.ufl_element(),
                           V_s.ufl_element(),
                           V_m.ufl_element()])
W = FunctionSpace(mesh, mixed_elem)               # (u_f, p_f, u_s, d_f)

# -----------------------------------------------------------------
# 4. Trial / test functions (mixed)
# -----------------------------------------------------------------
(u_f, p_f, u_s, d_f) = TrialFunctions(W)
(v_f, q_f, v_s, w_m) = TestFunctions(W)

# -----------------------------------------------------------------
# 5. Material parameters
# -----------------------------------------------------------------
rho_f = 1000.0
nu_f  = 1e-3
mu_f  = rho_f * nu_f

rho_s = 10000.0
nu_s  = 0.4
mu_s  = 0.5e6
lambda_s = 2.0 * mu_s * nu_s / (1.0 - 2.0 * nu_s)

# -----------------------------------------------------------------
# 6. Time stepping parameters
# -----------------------------------------------------------------
dt = 0.001
T  = 4.0
t  = 0.0

# -----------------------------------------------------------------
# 7. Kinematics (solid)
# -----------------------------------------------------------------
F_s = Identity(2) + grad(u_s)
E_s = 0.5 * (F_s.T * F_s - Identity(2))

# -----------------------------------------------------------------
# 8. Constitutive relations
# -----------------------------------------------------------------
def sigma_f(v, p):
    return -p * Identity(2) + 2.0 * mu_f * sym(grad(v))

def sigma_s():
    return lambda_s * tr(E_s) * Identity(2) + 2.0 * mu_s * E_s

# -----------------------------------------------------------------
# 9. Mesh smoothing (Laplace) – solve for d_f only
# -----------------------------------------------------------------
# Assemble on the mesh‑displacement space V_m
d_mesh, w_mesh = TrialFunction(V_m), TestFunction(V_m)
a_mesh = inner(grad(d_mesh), grad(w_mesh))*dx(1) \
         + 1e-6*inner(d_mesh, w_mesh)*dx(1)          # tiny mass term
L_mesh = dot(Constant((0.0, 0.0)), w_mesh)*dx(1)       # zero RHS

# Dirichlet BCs for the mesh displacement (on V_m)
bcd_inlet_m  = DirichletBC(V_m, Constant((0.0, 0.0)), boundaries, 1)
bcd_outlet_m = DirichletBC(V_m, Constant((0.0, 0.0)), boundaries, 2)
bcd_walls_m  = DirichletBC(V_m, Constant((0.0, 0.0)), boundaries, 3)
mesh_bcs = [bcd_inlet_m, bcd_outlet_m, bcd_walls_m]

# holder for the mesh displacement
d_f_sol = Function(V_m)

# -----------------------------------------------------------------
# 10. Fluid (steady Stokes on the current mesh)
# -----------------------------------------------------------------
a_fluid = (rho_f / dt) * inner(u_f, v_f) * dx(1) \
           + inner(sigma_f(u_f, p_f), grad(v_f)) * dx(1) \
           + q_f * div(u_f) * dx(1) \
           + p_f * q_f * dx(1)                     # PSPG stabilisation

# -----------------------------------------------------------------
# 11. Solid (dynamic St. Venant–Kirchhoff)
# -----------------------------------------------------------------
beta = 0.25
gamma = 0.5
a_solid = (rho_s / dt**2) * inner(u_s, v_s) * dx(2) \
           + inner(sigma_s(), grad(v_s)) * dx(2)

# -----------------------------------------------------------------
# 12. Coupling (traction balance on pole boundary)
# -----------------------------------------------------------------
n_f = FacetNormal(mesh)
traction = dot(sigma_f(u_f, p_f), n_f)
a_couple = dot(traction, v_s) * ds(4)

# -----------------------------------------------------------------
# 13. Monolithic bilinear form and RHS (zero RHS)
# -----------------------------------------------------------------
a = a_fluid + a_solid + a_couple
L = Constant(0.0) * dx(1) + Constant(0.0) * dx(2)   # zero linear form

# -----------------------------------------------------------------
# 14. Boundary conditions – inlet velocity (fixed)
# -----------------------------------------------------------------
U_bar = 1.0
H_chan = H

expr_inlet = Expression(
    ("factor*1.5*U_bar*x[1]*(H_chan - x[1])/(pow(H_chan/2.0,2))", "0.0"),
    factor=0.0,
    U_bar=U_bar,
    H_chan=H_chan,
    degree=2)

bcu_inlet = DirichletBC(W.sub(0), expr_inlet, boundaries, 1)
bcu_walls = DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 3)

bcd_inlet  = DirichletBC(W.sub(3), Constant((0.0, 0.0)), boundaries, 1)
bcd_outlet = DirichletBC(W.sub(3), Constant((0.0, 0.0)), boundaries, 2)
bcd_walls  = DirichletBC(W.sub(3), Constant((0.0, 0.0)), boundaries, 3)

bcs_solid = DirichletBC(W.sub(2), Constant((0.0, 0.0)), boundaries, 4)

bcs = [bcu_inlet, bcu_walls,
       bcd_inlet, bcd_outlet, bcd_walls,
       bcs_solid]

# -----------------------------------------------------------------
# 15. Time stepping
# -----------------------------------------------------------------
w = Function(W)          # (u_f, p_f, u_s, d_f) at new step
w0 = Function(W)         # previous step (kept for completeness)
(u_f0, p_f0, u_s0, d_f0) = w0.split(True)

# point‑A displacement file
file_A = open("point_A_disp.csv", "w")
file_A.write("t,dx,dy\n")

# XDMF files
xdmf_u = XDMFFile(mesh.mpi_comm(), "velocity.xdmf")
xdmf_p = XDMFFile(mesh.mpi_comm(), "pressure.xdmf")
xdmf_s = XDMFFile(mesh.mpi_comm(), "displacement.xdmf")
xdmf_u.parameters["flush_output"] = True
xdmf_p.parameters["flush_output"] = True
xdmf_s.parameters["flush_output"] = True

while t < T + DOLFIN_EPS:
    t += dt

    # ramp factor (t < 2 s) – update inlet profile
    factor = (1.0 - cos(pi * t / 2.0)) / 2.0 if t < 2.0 else 1.0
    expr_inlet.factor = factor

    # ----- mesh smoothing (solve for d_f alone) -----
    A_mesh = assemble(a_mesh)
    b_mesh = assemble(L_mesh)
    for bc in mesh_bcs:                     # only mesh BCs
        bc.apply(A_mesh, b_mesh)

    mesh_solver = PETScKrylovSolver("cg", "hypre_amg")
    mesh_solver.solve(d_f_sol.vector(), b_mesh)

    # copy mesh displacement into the mixed solution
    w.sub(3).vector()[:] = d_f_sol.vector()

    # ----- monolithic fluid–solid solve -----
    A = assemble(a)
    b = assemble(L)
    for bc in bcs:
        bc.apply(A, b)
    solve(A, w.vector(), b, "cg", "ilu")

    # ----- write output -----
    (u_f, p_f, u_s, d_f) = w.split(True)
    xdmf_u.write(u_f, t)
    xdmf_p.write(p_f, t)
    xdmf_s.write(u_s, t)

    # point A displacement
    A0 = Point(0.60, 0.20)
    disp_A = u_s(A0)
    file_A.write("{:.5f},{:.6e},{:.6e}\n".format(t, disp_A[0], disp_A[1]))

# ----- close files -----
file_A.close()
xdmf_u.close()
xdmf_p.close()
xdmf_s.close()