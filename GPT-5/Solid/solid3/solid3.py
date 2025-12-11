# q3_plate_with_hole_planes_tress.py
# Legacy FEniCS (dolfin + mshr)
#
# Problem:
#  - Geometry: (0,1.0) x (0,0.20) m with a centred circular hole of radius a=0.05 m at (0.50, 0.10)
#  - Model: Plane-stress linear elasticity
#  - Material: E=200e9 Pa, nu=0.30
#  - BCs:
#      * Left edge x=0: u=(0,0)
#      * Right edge x=1: traction t=(2e6, 0) N/m  [2 MPa * thickness 1 m]
#      * All other external boundaries (incl. hole): traction-free
#  - Outputs:
#      * von Mises (plane-stress) colour map -> q3_vm.png
#      * Displacement field -> q3_disp.xdmf
#      * Report sigma_max on hole boundary and Kt = sigma_max / (2 MPa)

from dolfin import *
from mshr import Rectangle, Circle, generate_mesh
import math

# -----------------------
# Parameters and geometry
# -----------------------
E  = 200e9
nu = 0.30

# Plane-stress parameters
mu = E / (2.0 * (1.0 + nu))                          # shear modulus
lam_ps = E * nu / (1.0 - nu**2)                      # effective lambda for plane-stress

# Domain sizes
Lx, Ly = 1.0, 0.20
xc, yc, a = 0.50, 0.10, 0.05

# Build CSG geometry and initial mesh
rect = Rectangle(Point(0.0, 0.0), Point(Lx, Ly))
hole = Circle(Point(xc, yc), a)
geom = rect - hole

# Base mesh; then locally refine near the hole for better stress resolution
mesh = generate_mesh(geom, 96)  # start with a reasonably fine mesh

# Local refinement around the hole (2 rounds)
for _ in range(2):
    cell_markers = MeshFunction("bool", mesh, mesh.topology().dim(), False)
    for c in cells(mesh):
        mp = c.midpoint()
        r = math.hypot(mp.x() - xc, mp.y() - yc)
        if r < a + 0.03:  # refine an annulus around the hole
            cell_markers[c] = True
    mesh = refine(mesh, cell_markers)

# -----------------------
# Function spaces
# -----------------------
V = VectorFunctionSpace(mesh, "CG", 2)   # quadratic for better stress resolution
u  = Function(V, name="displacement")
v  = TestFunction(V)
du = TrialFunction(V)

# -----------------------
# Boundary marking
# -----------------------
LEFT, RIGHT, HOLE = 1, 2, 3
facets = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)

class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, DOLFIN_EPS)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], Lx, DOLFIN_EPS)

class HoleBoundary(SubDomain):
    def inside(self, x, on_boundary):
        if not on_boundary:
            return False
        r = math.hypot(x[0]-xc, x[1]-yc)
        # Mesh approximates the circle; allow a tolerance band
        return abs(r - a) < 1.0e-3

Left().mark(facets, LEFT)
Right().mark(facets, RIGHT)
HoleBoundary().mark(facets, HOLE)

ds = Measure("ds", domain=mesh, subdomain_data=facets)

# -----------------------
# Kinematics and constitutive law (plane-stress)
# -----------------------
def eps(w):
    return sym(grad(w))

def sigma_ps(w):
    e = eps(w)
    return 2.0*mu*e + lam_ps*tr(e)*Identity(2)

# -----------------------
# Variational problem
# -----------------------
# Neumann traction on the right edge: 2 MPa * 1 m = 2e6 N/m (plane-stress line load)
t_right = Constant((2.0e6, 0.0))

a_form = inner(sigma_ps(du), eps(v))*dx
L_form = dot(t_right, v)*ds(RIGHT)  # traction-free elsewhere is natural (no term)

# Essential BC on left edge
bc_left = DirichletBC(V, Constant((0.0, 0.0)), facets, LEFT)

# Solve
u = Function(V, name="u")
solve(lhs(a_form) == rhs(L_form), u, bcs=[bc_left],
      solver_parameters={
          "linear_solver": "mumps"
      })

# -----------------------
# Post-processing: von Mises (plane-stress)
# -----------------------
# For plane-stress with sigma_zz = 0, von Mises reduces to:
# vm = sqrt(sxx^2 - sxx*syy + syy^2 + 3*sxy^2)
S = sigma_ps(u)
sxx = S[0, 0]
syy = S[1, 1]
sxy = S[0, 1]
vm_expr = sqrt(sxx**2 - sxx*syy + syy**2 + 3.0*sxy**2)

Vsig = FunctionSpace(mesh, "CG", 1)
vm = project(vm_expr, Vsig)  # smooth scalar field for visualisation
vm.rename("von_Mises", "von_Mises")

# -----------------------
# Extract sigma_max along the hole boundary
# -----------------------
mesh.init(1, 0)  # ensure facet-vertex connectivity
hole_vertices = set()
for f in facets.mesh().facets():
    if facets[f.index()] == HOLE:
        for vtx in vertices(f):
            hole_vertices.add(vtx.index())

coords = mesh.coordinates()
sigma_max = 0.0
for vid in hole_vertices:
    x = coords[vid]
    sigma_val = vm(x)  # evaluate projected von Mises at vertex
    if sigma_val > sigma_max:
        sigma_max = sigma_val

Kt = sigma_max / (2.0e6)  # 2 MPa reference

# -----------------------
# Save outputs
# -----------------------
# XDMF outputs (open in ParaView)
with XDMFFile(mesh.mpi_comm(), "q3_disp.xdmf") as xdmf:
    xdmf.write(u)

with XDMFFile(mesh.mpi_comm(), "q3_vm.xdmf") as xdmf:
    xdmf.write(vm)

# PNG colour map for quick glance (uses dolfin's matplotlib wrapper)
try:
    import matplotlib.pyplot as plt
    plt.figure()
    c = plot(vm, title="von Mises stress (plane-stress)")
    plt.colorbar(c)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig("q3_vm.png", dpi=300)
    plt.close()
except Exception as e:
    # Headless environments may not support plotting; XDMF is still available
    print("Matplotlib plotting failed (headless environment likely). Skipping PNG. Error:", e)

# -----------------------
# Report
# -----------------------
print("Max von Mises stress on hole boundary: {:.6e} Pa".format(sigma_max))
print("Stress Concentration Factor Kt (sigma_max / 2 MPa): {:.6f}".format(Kt))