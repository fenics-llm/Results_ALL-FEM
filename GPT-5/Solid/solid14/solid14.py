# q14_two_holes_neo_hookean.py
# Legacy FEniCS (dolfin 2019.1.0) + mshr
from __future__ import print_function
from dolfin import *
from mshr import *
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# --------- Geometry (m) ----------
Lx, Ly = 1.0, 0.20
a = 0.04
cx1, cy1 = 0.40, 0.10
cx2, cy2 = 0.60, 0.10

rect = Rectangle(Point(0.0, 0.0), Point(Lx, Ly))
hole1 = Circle(Point(cx1, cy1), a, 64)
hole2 = Circle(Point(cx2, cy2), a, 64)
domain = rect - hole1 - hole2

# Mesh
res = 96  # increase for finer results
mesh = generate_mesh(domain, res)

# --------- Boundary markers ----------
left_id, right_id, bottom_id, top_id, hole1_id, hole2_id = 1, 2, 3, 4, 10, 11
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)

tol = 1e-8
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, tol)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], Lx, tol)

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0, tol)

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], Ly, tol)

class Hole1(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near((x[0]-cx1)**2 + (x[1]-cy1)**2, a*a, 5e-6)

class Hole2(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near((x[0]-cx2)**2 + (x[1]-cy2)**2, a*a, 5e-6)

Left().mark(boundaries, left_id)
Right().mark(boundaries, right_id)
Bottom().mark(boundaries, bottom_id)
Top().mark(boundaries, top_id)
Hole1().mark(boundaries, hole1_id)
Hole2().mark(boundaries, hole2_id)

ds_ = Measure("ds", domain=mesh, subdomain_data=boundaries)

# --------- Function spaces (Taylor–Hood) ----------
Ve = VectorElement("Lagrange", mesh.ufl_cell(), 2)  # u
Qe = FiniteElement("Lagrange", mesh.ufl_cell(), 1)  # p
W = FunctionSpace(mesh, MixedElement([Ve, Qe]))

# Trial/Test and functions
w = Function(W)            # unknown (u, p)
dw = TrialFunction(W)
(v, q) = TestFunctions(W)
(u, p) = split(w)

# --------- Material parameters ----------
E = 5.0e6       # Pa
nu = 0.49
mu = E/(2.0*(1.0 + nu))            # shear modulus
kappa = E/(3.0*(1.0 - 2.0*nu))     # bulk modulus for mild stabilisation

# Plane-strain kinematics in 2D
d = mesh.geometry().dim()
I = Identity(d)
F = I + grad(u)
C = F.T*F
J = det(F)
Ic = tr(C)

# Isochoric invariant and 1st PK stress (deviatoric Neo-Hookean)
Jm23 = J**(-2.0/3.0)               # J^{-2/3}
P_iso = mu*Jm23*(F - (Ic/3.0)*inv(F).T)

# Volumetric via Lagrange multiplier p (hydrostatic)
P_vol = -p*J*inv(F).T

# Total 1st PK stress
P = P_iso + P_vol

# --------- Loading and boundary conditions ----------
# Prescribed displacements
ux_right = 0.060  # m
zero = Constant(0.0)
bc_left_u = DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, left_id)
bc_right_ux = DirichletBC(W.sub(0).sub(0), Constant(ux_right), boundaries, right_id)
bc_right_uy = DirichletBC(W.sub(0).sub(1), Constant(0.0), boundaries, right_id)
bcs = [bc_left_u, bc_right_ux, bc_right_uy]
# (no essential BC on p)

# Follower pressure on hole boundaries (current normal pulled back)
P_hole = Constant(0.10e6)  # Pa
N = FacetNormal(mesh)

# External virtual work (negative because traction is -P n)
Gext = -P_hole*dot(v, J*inv(F).T*N)*(ds_(hole1_id) + ds_(hole2_id))

# --------- Residual and Jacobian ----------
# Internal virtual work
Gint = inner(P, grad(v))*dx

# Mixed incompressibility constraint with mild pressure stabilisation
# -(J-1)*q + (p/kappa)*q = 0 enforces p ≈ kappa*(J-1)
Gcons = (-(J - 1.0)*q + (p/kappa)*q)*dx

R = Gint + Gcons + Gext
J_form = derivative(R, w, dw)

# --------- Nonlinear solve ----------
# Provide a sensible initial guess (zero)
assign(w.sub(0), project(Constant((0.0, 0.0)), W.sub(0).collapse()))
assign(w.sub(1), project(Constant(0.0), W.sub(1).collapse()))

solver_params = {
    "nonlinear_solver": "newton",
    "newton_solver": {
        "absolute_tolerance": 1e-8,
        "relative_tolerance": 1e-6,
        "maximum_iterations": 40,
        "linear_solver": "mumps",
        "report": True
    }
}
solve(R == 0, w, bcs, J=J_form, solver_parameters=solver_params)

(u, p) = w.split(deepcopy=True)

# --------- Cauchy stress and von Mises (plane-strain 2D measure) ----------
# Cauchy stress: sigma = (1/J) * P * F^T  (2×2 here)
sigma = (1.0/J)*P*F.T
sigma = variable(sigma)  # for UFL safety (though not strictly required)

# Components
sxx = sigma[0, 0]
syy = sigma[1, 1]
sxy = sigma[0, 1]

# 2D plane-strain von Mises measure:
#   vm = sqrt( sxx^2 - sxx*syy + syy^2 + 3*sxy^2 )
vm_expr = sqrt(sxx**2 - sxx*syy + syy**2 + 3.0*sxy**2)

Q1 = FunctionSpace(mesh, "Lagrange", 1)
p_plot = project(p, Q1)
vm = project(vm_expr, Q1)

# --------- Save fields ----------
# Displacement to XDMF
xdmf = XDMFFile(mesh.mpi_comm(), "q14_displacement.xdmf")
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True
xdmf.write(u)

# Also save pressure and von Mises as XDMF (optional but handy)
xdmf_p = XDMFFile(mesh.mpi_comm(), "q14_pressure.xdmf")
xdmf_vm = XDMFFile(mesh.mpi_comm(), "q14_vonmises.xdmf")
xdmf_p.parameters["flush_output"] = True
xdmf_vm.parameters["flush_output"] = True
xdmf_p.parameters["functions_share_mesh"] = True
xdmf_vm.parameters["functions_share_mesh"] = True
xdmf_p.write(p_plot)
xdmf_vm.write(vm)

# --------- Static colour maps (PNG) ----------
# Pressure
plt.figure(figsize=(6.8, 1.6))
c = plot(p_plot, title="Pressure p (Pa)")
plt.colorbar(c)
plt.axis("off")
plt.tight_layout()
plt.savefig("q14_p.png", dpi=220)
plt.close()

# von Mises
plt.figure(figsize=(6.8, 1.6))
c = plot(vm, title="von Mises (Pa)")
plt.colorbar(c)
plt.axis("off")
plt.tight_layout()
plt.savefig("q14_vm.png", dpi=220)
plt.close()

print("Saved: q14_p.png, q14_vm.png, q14_displacement.xdmf (plus auxiliary XDMFs).")