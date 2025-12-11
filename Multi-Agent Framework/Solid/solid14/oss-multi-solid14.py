# -*- coding: utf-8 -*-
#
# Large-deformation Neo-Hookean strip with two holes (legacy FEniCS)
#
#   Ω = (0,1)×(0,0.20) \ (two circles of radius a=0.04 centred at (0.40,0.10) and (0.60,0.10))
#
#   Material:  E = 5 MPa, ν = 0.49   (plane strain)
#   BCs:
#       left  (x=0)          : u = (0,0)
#       right (x=1)          : u = (0.060,0)
#       holes (r=0.04)       : follower pressure P_hole = 0.10 MPa
#       top/bottom (y=0,0.20): traction free
#
#   Unknowns: displacement u = (u_x,u_y) and pressure p
#   Outputs: pressure PNG (q14_p.png), von Mises PNG (q14_vm.png), displacement XDMF (q14_disp.xdmf)
#
# --------------------------------------------------------------
from dolfin import *
import mshr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ufl_legacy import inv               # <-- correct import for legacy FEniCS

# --------------------------------------------------------------
# 1. Geometry & mesh
# --------------------------------------------------------------
L, H = 1.0, 0.20
a = 0.04
c1 = Point(0.40, 0.10)
c2 = Point(0.60, 0.10)

domain = mshr.Rectangle(Point(0.0, 0.0), Point(L, H)) \
         - mshr.Circle(c1, a) \
         - mshr.Circle(c2, a)

mesh = mshr.generate_mesh(domain, 80)   # refine if needed

# --------------------------------------------------------------
# 2. Function spaces (Taylor–Hood)
# --------------------------------------------------------------
Ve = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Pe = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = MixedElement([Ve, Pe])
W  = FunctionSpace(mesh, TH)

# --------------------------------------------------------------
# 3. Boundary markers
# --------------------------------------------------------------
tol = 1E-8

class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, tol)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], L, tol)

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], H, tol)

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0, tol)

class Hole1(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and ((x[0]-c1.x())**2 + (x[1]-c1.y())**2 < (a+tol)**2)

class Hole2(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and ((x[0]-c2.x())**2 + (x[1]-c2.y())**2 < (a+tol)**2)

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
Left().mark(boundaries, 1)
Right().mark(boundaries, 2)
Top().mark(boundaries, 3)
Bottom().mark(boundaries, 4)
Hole1().mark(boundaries, 5)
Hole2().mark(boundaries, 6)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# --------------------------------------------------------------
# 4. Material parameters (plane strain)
# --------------------------------------------------------------
E, nu = 5e6, 0.49
mu    = E/(2.0*(1.0+nu))
lmbda = E*nu/((1.0+nu)*(1.0-2.0*nu))

# --------------------------------------------------------------
# 5. Mixed trial/test functions
# --------------------------------------------------------------
w = Function(W)               # (u,p)
(u, p) = split(w)
(v, q) = TestFunctions(W)

# --------------------------------------------------------------
# 6. Kinematics
# --------------------------------------------------------------
I  = Identity(2)
F  = I + grad(u)              # deformation gradient
J  = det(F)
C  = F.T*F
Ic = tr(C)

# --------------------------------------------------------------
# 7. Neo-Hookean stress (first Piola–Kirchhoff)
# --------------------------------------------------------------
P_iso = mu*J**(-2.0/3.0)*(F - (1.0/3.0)*Ic*inv(F).T)
P_vol = p*J*inv(F).T
P     = P_iso + P_vol

# --------------------------------------------------------------
# 8. External loads
# --------------------------------------------------------------
P_hole = 0.10e6                # 0.10 MPa
T_hole = -P_hole * J * inv(F).T * FacetNormal(mesh)   # follower pressure (Piola traction)

# --------------------------------------------------------------
# 9. Dirichlet BCs
# --------------------------------------------------------------
bc_left  = DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 1)
u_right = Expression(("0.060", "0.0"), degree=1)
bc_right = DirichletBC(W.sub(0), u_right, boundaries, 2)

# pressure gauge – fix p on the left edge (any point on that edge suffices)
bc_p = DirichletBC(W.sub(1), Constant(0.0), "near(x[0], 0.0) && near(x[1], 0.0)", "pointwise")

bcs = [bc_left, bc_right, bc_p]

# --------------------------------------------------------------
# 10. Variational form (weak equilibrium)
# --------------------------------------------------------------
F_int = inner(P, grad(v))*dx
F_vol = q*(J - 1.0)*dx
F_ext = dot(T_hole, v)*(ds(5) + ds(6))
F = F_int + F_vol - F_ext

# --------------------------------------------------------------
# 11. Solve (Newton)
# --------------------------------------------------------------
solve(F == 0, w, bcs,
      solver_parameters={"newton_solver":
        {"relative_tolerance":1e-6, "absolute_tolerance":1e-8, "maximum_iterations":30}})

# --------------------------------------------------------------
# 12. Split solution
# --------------------------------------------------------------
(u_h, p_h) = w.split(deepcopy=True)

# --------------------------------------------------------------
# 13. Cauchy stress & von Mises
# --------------------------------------------------------------
F_h = I + grad(u_h)
J_h = det(F_h)
P_h = mu*J_h**(-2.0/3.0)*(F_h - (1.0/3.0)*tr(F_h.T*F_h)*inv(F_h).T) + p_h*J_h*inv(F_h).T
sigma = (1.0/J_h)*P_h*F_h.T
s = sigma - (1.0/3.0)*tr(sigma)*I
vonMises = sqrt(3.0/2.0*inner(s, s))

p_proj = project(p_h, FunctionSpace(mesh, "Lagrange", 1))
vm_proj = project(vonMises, FunctionSpace(mesh, "Lagrange", 1))

# --------------------------------------------------------------
# 14. Output
# --------------------------------------------------------------
with XDMFFile(mesh.mpi_comm(), "q14_disp.xdmf") as xdmf:
    xdmf.write(u_h, 0.0)

plt.figure()
p_plot = plot(p_proj, title="Hydrostatic pressure (Pa)", cmap="viridis")
plt.colorbar(p_plot)
plt.savefig("q14_p.png", dpi=300, bbox_inches='tight')
plt.close()

plt.figure()
vm_plot = plot(vm_proj, title="Von Mises stress (Pa)", cmap="viridis")
plt.colorbar(vm_plot)
plt.savefig("q14_vm.png", dpi=300, bbox_inches='tight')
plt.close()