# q4_two_holes_plane_stress.py
from dolfin import *
from mshr import Rectangle, Circle, generate_mesh
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

# ----------------------------
# Material (plane stress)
# ----------------------------
E  = 200e9         # Young's modulus in Pa (200 GPa)
nu = 0.30          # Poisson's ratio
mu = E/(2.0*(1.0+nu))
# Plane-stress effective lambda:
lam = E*nu/(1.0 - nu**2)

# ----------------------------
# Geometry (metres)
# ----------------------------
L, H = 1.0, 0.20
a = 0.04
c1 = Point(0.33, 0.10)
c2 = Point(0.67, 0.10)

# Construct CSG domain: rectangle minus two circles
domain = Rectangle(Point(0.0, 0.0), Point(L, H)) \
         - Circle(c1, a, segments=96) \
         - Circle(c2, a, segments=96)

# Mesh (increase resolution if needed)
mesh = generate_mesh(domain, 160)  # ~fine enough for stress near holes

# ----------------------------
# Boundary markers
# ----------------------------
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, DOLFIN_EPS)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], L, DOLFIN_EPS)

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0, DOLFIN_EPS)

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], H, DOLFIN_EPS)

class Hole1(SubDomain):
    def inside(self, x, on_boundary):
        # circle (x - xc)^2 + (y - yc)^2 = a^2
        return on_boundary and near((x[0]-c1.x())**2 + (x[1]-c1.y())**2, a**2, 5e-4)

class Hole2(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near((x[0]-c2.x())**2 + (x[1]-c2.y())**2, a**2, 5e-4)

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
Left().mark(boundaries,   1)
Right().mark(boundaries,  2)
Bottom().mark(boundaries, 3)
Top().mark(boundaries,    4)
Hole1().mark(boundaries,  5)
Hole2().mark(boundaries,  6)

ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

# ----------------------------
# Function spaces
# ----------------------------
V = VectorFunctionSpace(mesh, "CG", 2)

# Dirichlet BC: Left edge clamped u=(0,0)
u0 = Constant((0.0, 0.0))
bc_left = DirichletBC(V, u0, boundaries, 1)
bcs = [bc_left]

# Neumann traction on right edge (2 MPa along +x)
t_right = Constant((2.0e6, 0.0))  # Pa

# ----------------------------
# Variational problem (plane-stress)
# ----------------------------
u = TrialFunction(V)
v = TestFunction(V)
I = Identity(2)
def eps(w):
    return sym(grad(w))
def sigma_ps(w):
    e = eps(w)
    return 2.0*mu*e + lam*tr(e)*I

a = inner(sigma_ps(u), eps(v))*dx
L = dot(t_right, v)*ds(2)  # only on right boundary

# Solve
u_sol = Function(V, name="displacement")
solve(a == L, u_sol, bcs, solver_parameters={"linear_solver": "mumps"})

# ----------------------------
# Post-processing: von Mises (plane stress)
# ----------------------------
W = TensorFunctionSpace(mesh, "DG", 0)
S = sigma_ps(u_sol)
S_proj = project(S, W)  # Cauchy stress tensor field (piecewise-constant)

# von Mises for plane stress: sqrt(sxx^2 + syy^2 - sxx*syy + 3*tauxy^2)
Q = FunctionSpace(mesh, "CG", 1)
sxx = project(S_proj.sub(0), Q)  # note: better compute from S directly below
# safer: compute directly as UFL expression then project once
Sx = S_proj  # alias

# Build UFL scalars from S (not from projected components to avoid ordering issues)
S_ufl = S  # UFL tensor
s_xx = S_ufl[0,0]
s_yy = S_ufl[1,1]
s_xy = S_ufl[0,1]

vm_expr = sqrt(s_xx**2 + s_yy**2 - s_xx*s_yy + 3.0*s_xy**2)
vm = project(vm_expr, Q, solver_type="cg", preconditioner_type="hypre_amg")
vm.rename("von_Mises", "von_Mises")

# ----------------------------
# Find max von Mises on hole boundaries (IDs 5 and 6)
# ----------------------------
# Collect vertices that belong to facets marked as holes
mesh.init(1, 0)  # ensure facet->vertex connectivity
hole_vertex_ids = set()
for facet in facets(mesh):
    m = boundaries[facet]
    if m in (5, 6):
        for vid in facet.entities(0):
            hole_vertex_ids.add(vid)

coords = mesh.coordinates()
vm_array = vm.compute_vertex_values(mesh)
sigmax = -1.0
for vid in hole_vertex_ids:
    val = vm_array[vid]
    if val > sigmax:
        sigmax = val

# Stress Concentration Factor with reference = 2 MPa
Kt = sigmax / 2.0e6

# ----------------------------
# Save outputs
# ----------------------------
# 1) Displacement to XDMF
with XDMFFile(mesh.mpi_comm(), "q4_displacement.xdmf") as xdmf:
    xdmf.parameters["flush_output"] = True
    xdmf.parameters["functions_share_mesh"] = True
    xdmf.write(u_sol)

# 2) von Mises colour map as PNG
# Interpolate vm to CG1 and plot with tripcolor
vm_cg1 = vm  # already CG1
# Build a Triangulation for tripcolor
coor = mesh.coordinates()
cells = mesh.cells()
vals = vm_cg1.compute_vertex_values(mesh)

plt.figure(figsize=(8, 2))
# Note: tripcolor expects triangles; FEniCS tetra-to-tri cells are fine in 2D
import matplotlib.tri as mtri
triang = mtri.Triangulation(coor[:,0], coor[:,1], cells)
tpc = plt.tripcolor(triang, vals, shading='gouraud')
plt.colorbar(tpc, label="von Mises stress (Pa)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("von Mises equivalent stress (plane stress)")
plt.tight_layout()
plt.savefig("q4_vm.png", dpi=300)
plt.close()

# 3) Print report
rank = MPI.rank(mesh.mpi_comm())
if rank == 0:
    print("=== Report (two-hole plate, plane stress) ===")
    print("E = %.3e Pa, nu = %.3f" % (E, nu))
    print("mu = %.3e Pa, lambda_ps = %.3e Pa" % (mu, lam))
    print("Right-edge traction = (%.3e, %.3e) Pa" % (t_right(0.0)[0], t_right(0.0)[1]))
    print("Max von Mises on hole boundaries: %.6e Pa" % sigmax)
    print("Stress Concentration Factor Kt = sigma_max / 2 MPa = %.6f" % Kt)
    print("Saved: q4_displacement.xdmf, q4_vm.png")

# Optional: also save vm field to XDMF for Paraview
with XDMFFile(mesh.mpi_comm(), "q4_vm.xdmf") as xdmf_vm:
    xdmf_vm.parameters["flush_output"] = True
    xdmf_vm.parameters["functions_share_mesh"] = True
    xdmf_vm.write(vm)