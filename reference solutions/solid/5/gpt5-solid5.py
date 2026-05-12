from dolfin import *
from mshr import *
import matplotlib.pyplot as plt

# --- Geometry (metres) ---
Lx = 1.0
Ly = 0.20
notch_x = 0.06
notch_y1, notch_y2 = 0.08, 0.12

# Outer rectangle minus notch cut-out
outer = Rectangle(Point(0.0, 0.0), Point(Lx, Ly))
notch = Rectangle(Point(0.0, notch_y1), Point(notch_x, notch_y2))
domain = outer - notch

# Mesh (increase resolution if you like)
mesh = generate_mesh(domain, 128)

# --- Function space ---
V = VectorFunctionSpace(mesh, "CG", 2)

# --- Material (plane stress) ---
E  = 200e9         # Pa
nu = 0.30
mu = E/(2.0*(1.0 + nu))
lam_ps = 2.0*mu*nu/(1.0 - nu)   # plane-stress effective lambda

def eps(u):
    return sym(grad(u))

def sigma(u):
    return 2.0*mu*eps(u) + lam_ps*tr(eps(u))*Identity(2)

# --- Boundary markers for Neumann traction ---
tol = 1e-8

class LeftFixed(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, tol)

class RightEdge(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], Lx, tol)

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
boundaries.set_all(0)
LeftFixed().mark(boundaries, 10)
RightEdge().mark(boundaries, 1)

dsN = Measure("ds", domain=mesh, subdomain_data=boundaries)

# --- Boundary conditions ---
u0 = Constant((0.0, 0.0))
bcs = [DirichletBC(V, u0, boundaries, 10)]  # x=0 fixed (u=0)

# --- Loads (traction on right edge) ---
# Given: sigma*n = (2 MPa, 0)    (units of traction, i.e., N/m^2)
T = Constant((2.0e6, 0.0))

# --- Variational problem ---
u = TrialFunction(V)
v = TestFunction(V)
a = inner(sigma(u), eps(v))*dx
L = dot(T, v)*dsN(1)  # only on right edge; others are traction-free

# --- Solve ---
u_sol = Function(V, name="displacement")
solve(a == L, u_sol, bcs,
      solver_parameters={"linear_solver": "mumps"} if has_linear_algebra_backend("PETSc") else {})

# --- Post-process: von Mises (plane stress) ---
# sigma(u) is 2x2; von Mises (plane stress): sqrt(sxx^2 - sxx*syy + syy^2 + 3*sxy^2)
sig = sigma(u_sol)
sxx = sig[0, 0]
syy = sig[1, 1]
sxy = sig[0, 1]
vm_expr = sqrt(sxx**2 - sxx*syy + syy**2 + 3.0*sxy**2)

VM = project(vm_expr, FunctionSpace(mesh, "CG", 1), solver_type="cg", preconditioner_type="icc")
VM.rename("von_Mises", "von_Mises")

# --- Save displacement to XDMF ---
with XDMFFile(mesh.mpi_comm(), "displacement.xdmf") as xdmf:
    xdmf.parameters["functions_share_mesh"] = True
    xdmf.parameters["flush_output"] = True
    xdmf.write(mesh)
    xdmf.write(u_sol)

# (Optional) also save VM to XDMF for post-processing
with XDMFFile(mesh.mpi_comm(), "von_mises.xdmf") as xdmf_vm:
    xdmf_vm.parameters["functions_share_mesh"] = True
    xdmf_vm.parameters["flush_output"] = True
    xdmf_vm.write(VM)

# --- Save a colour map PNG of von Mises ---
plt.figure()
p = plot(VM, title="von Mises stress (Pa)")  # legacy FEniCS plot
plt.colorbar(p)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.tight_layout()
plt.savefig("q5_vm.png", dpi=200)
plt.close()

print("Done. Wrote displacement.xdmf, von_mises.xdmf, and q5_vm.png")