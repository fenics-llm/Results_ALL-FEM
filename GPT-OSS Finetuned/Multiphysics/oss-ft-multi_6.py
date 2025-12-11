# filename: flag_fsi_legacy.py
from fenics import *
from mshr import *
import numpy as np

# -------------------------------------------------
# Geometry
L, H = 2.5, 0.41
flag_len, flag_thk = 0.35, 0.02
pole_center = Point(0.20, 0.20)
pole_radius = 0.05

channel = Rectangle(Point(0.0, 0.0), Point(L, H))
pole    = Circle(pole_center, pole_radius, 64)

fluid_geom = channel - pole
flag_origin = Point(0.60, 0.19)
flag = Rectangle(flag_origin,
                 Point(flag_origin.x() + flag_len,
                       flag_origin.y() + flag_thk))
solid_geom = flag - pole

# Mesh
mesh = generate_mesh(fluid_geom + solid_geom, 80)

# -------------------------------------------------
# Subdomains (cell markers)
class FluidDomain(SubDomain):
    def inside(self, x, on_boundary):
        return (x[0] >= -DOLFIN_EPS) and (x[0] <= L + DOLFIN_EPS) and \
               (x[1] >= -DOLFIN_EPS) and (x[1] <= H + DOLFIN_EPS) and \
               ((x[0] - pole_center.x())**2 + (x[1] - pole_center.y())**2 >= pole_radius**2 - DOLFIN_EPS)

class SolidDomain(SubDomain):
    def inside(self, x, on_boundary):
        return (x[0] >= flag_origin.x() - DOLFIN_EPS) and (x[0] <= flag_origin.x() + flag_len + DOLFIN_EPS) and \
               (x[1] >= flag_origin.y() - DOLFIN_EPS) and (x[1] <= flag_origin.y() + flag_thk + DOLFIN_EPS) and \
               ((x[0] - pole_center.x())**2 + (x[1] - pole_center.y())**2 >= pole_radius**2 - DOLFIN_EPS)

cell_markers = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
FluidDomain().mark(cell_markers, 1)
SolidDomain().mark(cell_markers, 2)

dx = Measure('dx', domain=mesh, subdomain_data=cell_markers)
dx_f = dx(1)
dx_s = dx(2)

# -------------------------------------------------
# Boundary facets
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)

class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and on_boundary
class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], L) and on_boundary
class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[1], 0.0) or near(x[1], H)) and on_boundary
class FlagBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = DOLFIN_EPS
        on_flag = (near(x[0], flag_origin.x()) or near(x[0], flag_origin.x() + flag_len) or
                   near(x[1], flag_origin.y()) or near(x[1], flag_origin.y() + flag_thk))
        not_in_pole = ((x[0] - pole_center.x())**2 + (x[1] - pole_center.y())**2 >= pole_radius**2 - tol)
        return on_flag and not_in_pole and on_boundary
class PoleBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = DOLFIN_EPS
        in_pole = ((x[0] - pole_center.x())**2 + (x[1] - pole_center.y())**2 <= pole_radius**2 + tol)
        return in_pole and on_boundary

Inlet().mark(boundaries, 1)
Outlet().mark(boundaries, 2)
Walls().mark(boundaries, 3)
FlagBoundary().mark(boundaries, 4)
PoleBoundary().mark(boundaries, 5)

ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# -------------------------------------------------
# Mixed function space (fluid velocity, pressure, solid displacement)
V_f = VectorFunctionSpace(mesh, "CG", 2)
Q_f = FunctionSpace(mesh, "CG", 1)
V_s = VectorFunctionSpace(mesh, "CG", 2)

ME = MixedElement([V_f.ufl_element(), Q_f.ufl_element(), V_s.ufl_element()])
W  = FunctionSpace(mesh, ME)

(u, p, d) = TrialFunctions(W)
(v, q, e) = TestFunctions(W)

# -------------------------------------------------
# Material parameters
rho_f = 1000.0
nu_f = 1.0e-3
mu_f = rho_f * nu_f

rho_s = 10000.0
nu_s = 0.4
E_s = 0.5e6
mu_s = E_s / (2.0 * (1.0 + nu_s))
lmbda_s = E_s * nu_s / ((1.0 + nu_s) * (1.0 - 2.0 * nu_s))

# -------------------------------------------------
# Kinematics
I = Identity(2)
F = I + grad(d)               # deformation gradient
C = F.T * F
E_green = 0.5 * (C - I)      # Green-Lagrange strain

S_s = lmbda_s * tr(E_green) * I + 2.0 * mu_s * E_green
P_s = F * S_s                 # 1st Piola–Kirchhoff stress

def epsilon(u):
    return sym(grad(u))
sigma_f = -p * I + 2.0 * mu_f * epsilon(u)

# -------------------------------------------------
# Time stepping
dt = 0.01
T  = 5.0
t  = 0.0

# Inlet velocity (time dependent)
U_bar = 1.0
inlet_expr = Expression(('Ufac*4.0*x[1]*(H - x[1])/H/H', '0.0'),
                        Ufac=0.0, H=H, degree=2)

# -------------------------------------------------
# Boundary conditions
bcu_inlet = DirichletBC(W.sub(0), inlet_expr, boundaries, 1)
bcu_walls = DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 3)
bcd_flag   = DirichletBC(W.sub(2), Constant((0.0, 0.0)), boundaries, 5)  # pole fixed
bcs = [bcu_inlet, bcu_walls, bcd_flag]

# -------------------------------------------------
# Initial condition
w   = Function(W)          # solution at new time level
w_n = Function(W)          # solution at previous time level
w_n.vector()[:] = 0.0

# split previous solution for use in forms
(u_n, p_n, d_n) = split(w_n)

# -------------------------------------------------
# Weak form (implicit Euler)
conv = dot(u_n, nabla_grad(u_n))
F_fluid = rho_f * dot((u - u_n) / dt, v) * dx_f \
          + rho_f * dot(conv, v) * dx_f \
          + inner(sigma_f, epsilon(v)) * dx_f

F_solid = rho_s * dot((d - d_n) / dt, e) * dx_s \
          + inner(P_s, grad(e)) * dx_s

F_cont = q * div(u) * dx_f

F = F_fluid + F_solid + F_cont

# -------------------------------------------------
# Solver
problem = LinearVariationalProblem(lhs(F), rhs(F), w, bcs)
solver  = LinearVariationalSolver(problem)

# -------------------------------------------------
# Output
xdmf_u = XDMFFile("fluid_velocity.xdmf")
xdmf_p = XDMFFile("fluid_pressure.xdmf")
xdmf_d = XDMFFile("solid_displacement.xdmf")
for f in (xdmf_u, xdmf_p, xdmf_d):
    f.parameters["flush_output"] = True

# -------------------------------------------------
# Record displacement of point A
A0 = np.array([0.60, 0.20])
A_file = open("pointA_displacement.txt", "w")
A_file.write("# t   d_x   d_y\n")

while t <= T + DOLFIN_EPS:
    # Update inlet factor
    if t < 2.0:
        fac = (1.0 - np.cos(np.pi * t / 2.0)) / 2.0
    else:
        fac = 1.0
    inlet_expr.Ufac = fac * U_bar

    # Solve for current step
    solver.solve()

    # Split solution
    (u_sol, p_sol, d_sol) = w.split()

    # Move mesh according to solid displacement (incremental)
    ALE.move(mesh, d_sol - d_n)

    # Update previous solution
    w_n.assign(w)

    # Save fields
    xdmf_u.write(u_sol, t)
    xdmf_p.write(p_sol, t)
    xdmf_d.write(d_sol, t)

    # Record point A displacement
    dA = d_sol(A0)
    A_file.write(f"{t:.4f} {dA[0]:.6e} {dA[1]:.6e}\n")

    t += dt

A_file.close()