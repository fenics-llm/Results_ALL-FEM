# FEniCS legacy script (tested with dolfin 2019.1.0, mshr)
from __future__ import print_function
from dolfin import *
from mshr import *
import numpy as np

# ---------- Geometry and mesh ----------
Lx, Ly = 1.0, 0.20         # metres
cx, cy = 0.50, 0.20        # centre of notch (metres)
a = 0.05                   # notch radius (metres)

rectangle = Rectangle(Point(0.0, 0.0), Point(Lx, Ly))
circle = Circle(Point(cx, cy), a, 64)
domain = rectangle - circle

# Mesh resolution (increase if you want a finer picture)
mesh = generate_mesh(domain, 128)

# ---------- Material (plane-stress) ----------
E = 200e9                  # Pa
nu = 0.30
mu = E / (2.0*(1.0 + nu))                  # shear modulus
lam_ps = (2.0*mu*nu) / (1.0 - nu)          # plane-stress lambda

# ---------- Function spaces ----------
V = VectorFunctionSpace(mesh, "CG", 2)     # quadratic for better stress
u = TrialFunction(V)
v = TestFunction(V)

# ---------- Boundary markers ----------
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
boundaries.set_all(0)

tol = 1e-8

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0, DOLFIN_EPS)

class TopLoaded(SubDomain):
    def inside(self, x, on_boundary):
        # Straight top edge at y = Ly, excluding x in [0.45, 0.55]
        return (on_boundary and near(x[1], Ly, DOLFIN_EPS) and
                (x[0] < 0.45 - tol or x[0] > 0.55 + tol))

Bottom().mark(boundaries, 1)
TopLoaded().mark(boundaries, 2)

ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

# ---------- Kinematics and constitutive law (plane-stress) ----------
def eps(w):
    return sym(grad(w))

def sigma(w):
    return 2.0*mu*eps(w) + lam_ps*tr(eps(w))*Identity(2)

# ---------- Loads ----------
# Given traction: sigma*n = (0, -10 MPa) on the two straight top segments
t = Constant((0.0, -10e6))  # N/m^2 (Pa)

# ---------- Variational problem ----------
a_form = inner(sigma(u), eps(v))*dx
L_form = dot(t, v)*ds(2)    # only on the marked top segments

# ---------- Essential boundary conditions ----------
zero_vec = Constant((0.0, 0.0))
bc_bottom = DirichletBC(V, zero_vec, boundaries, 1)
bcs = [bc_bottom]

# ---------- Solve ----------
u_sol = Function(V, name="displacement")
solve(a_form == L_form, u_sol, bcs, solver_parameters={"linear_solver": "mumps"})

# ---------- Post-processing: von Mises (plane-stress) ----------
# For plane stress: sigma_vm = sqrt(sxx^2 - sxx*syy + syy^2 + 3*sxy^2)
W = TensorFunctionSpace(mesh, "DG", 0)  # piecewise-constant stress
sigma_expr = project(sigma(u_sol), W)

sxx = sigma_expr[:, 0, 0]
syy = sigma_expr[:, 1, 1]
sxy = sigma_expr[:, 0, 1]
vm_expr = sqrt(sxx**2 - sxx*syy + syy**2 + 3.0*sxy**2)

VM = Function(FunctionSpace(mesh, "DG", 0), name="von_Mises")
VM.vector()[:] = vm_expr.vector()[:]

# ---------- Save outputs ----------
# 1) Displacement field in XDMF
with XDMFFile(mesh.mpi_comm(), "displacement.xdmf") as xdmf:
    xdmf.write(mesh)
    xdmf.write(u_sol)

# 2) Von Mises PNG colour map
try:
    import matplotlib.pyplot as plt
    # Sample VM at cell midpoints for a crisp DG(0) plot
    c = VM.compute_vertex_values(mesh)  # will broadcast to vertices; fine for a quick plot
    plt.figure()
    p = plot(VM, title="von Mises stress (Pa)")
    plt.colorbar(p)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.tight_layout()
    plt.savefig("q6_vm.png", dpi=300)
    plt.close()
    print("Saved q6_vm.png")
except Exception as e:
    print("Matplotlib plot failed:", e)

# Also save VM to XDMF for convenience
with XDMFFile(mesh.mpi_comm(), "von_mises.xdmf") as xdmf_vm:
    xdmf_vm.write(mesh)
    xdmf_vm.write(VM)

# ---------- Minimal console outputs ----------
Vmin = VM.vector().min()
Vmax = VM.vector().max()
disp_norm = norm(u_sol, 'L2')
if MPI.rank(mesh.mpi_comm()) == 0:
    print("||u||_L2 =", disp_norm)
    print("von Mises min/max (Pa):", Vmin, Vmax)