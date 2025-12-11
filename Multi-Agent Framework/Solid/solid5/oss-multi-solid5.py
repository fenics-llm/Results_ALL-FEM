# -*- coding: utf-8 -*-
#
# Plane-stress linear elasticity on a rectangular plate with a left notch.
# Left outer edge fixed, right edge loaded with 2 MPa·m traction.
# Outputs: displacement (XDMF) and von Mises stress (PNG).
#
# Legacy FEniCS (dolfin) implementation.
#
from dolfin import *
from mshr import Rectangle, generate_mesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# 1. Geometry and mesh
# --------------------------------------------------------------
L, H = 1.0, 0.20          # plate dimensions
a, b = 0.06, 0.04         # notch dimensions (centered vertically)

# Main rectangle
plate = Rectangle(Point(0.0, 0.0), Point(L, H))
# Notch rectangle (cut from the left side)
notch  = Rectangle(Point(0.0, 0.5*H - 0.5*b),
                    Point(a, 0.5*H + 0.5*b))
# Domain with notch removed
domain = plate - notch

# Mesh resolution (roughly 0.005 m cells)
mesh = generate_mesh(domain, 200)

# --------------------------------------------------------------
# 2. Function space (pure elasticity)
# --------------------------------------------------------------
V = VectorFunctionSpace(mesh, "Lagrange", 2)

# --------------------------------------------------------------
# 3. Material parameters (plane stress)
# --------------------------------------------------------------
E, nu = 2.0e11, 0.30
mu    = E/(2.0*(1.0 + nu))
lmbda = E*nu/(1.0 - nu**2)

# --------------------------------------------------------------
# 4. Boundary definitions
# --------------------------------------------------------------
tol = 1E-14

class LeftOuter(SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary and near(x[0], 0.0, tol) and
                (x[1] < 0.5*H - 0.5*b - tol or x[1] > 0.5*H + 0.5*b + tol))

class RightEdge(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], L, tol)

# Mark boundaries
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
left_outer = LeftOuter()
right_edge = RightEdge()
left_outer.mark(boundaries, 1)
right_edge.mark(boundaries, 2)

ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# --------------------------------------------------------------
# 5. Variational problem
# --------------------------------------------------------------
u = TrialFunction(V)
v = TestFunction(V)

def epsilon(u):
    return sym(grad(u))

def sigma(u):
    return lmbda*tr(epsilon(u))*Identity(2) + 2.0*mu*epsilon(u)

# Traction on the right edge (2 MPa·m in x-direction)
t = Constant((2.0e6, 0.0))

a = inner(sigma(u), epsilon(v))*dx
L_form = dot(t, v)*ds(2)   # only right edge

# Dirichlet BC on left outer edge
zero = Constant((0.0, 0.0))
bc = DirichletBC(V, zero, left_outer)

# --------------------------------------------------------------
# 6. Solve
# --------------------------------------------------------------
u_sol = Function(V, name="Displacement")
solve(a == L_form, u_sol, bc, solver_parameters={"linear_solver": "mumps"})

# --------------------------------------------------------------
# 7. Post-processing – von Mises stress
# --------------------------------------------------------------
s = sigma(u_sol)
s_vm = sqrt(s[0, 0]**2 - s[0, 0]*s[1, 1] + s[1, 1]**2 + 3.0*s[0, 1]**2)

# Project von Mises onto a scalar space for plotting
V_scalar = FunctionSpace(mesh, "Lagrange", 1)
s_vm_proj = project(s_vm, V_scalar, solver_type="mumps")

# Save displacement (XDMF)
with XDMFFile(mesh.mpi_comm(), "q5_disp.xdmf") as xdmf:
    xdmf.write(u_sol)

# Plot von Mises and save as PNG
plt.figure(figsize=(8, 3))
p = plot(s_vm_proj, title="von Mises stress (Pa)", cmap="viridis")
plt.colorbar(p)
plt.tight_layout()
plt.savefig("q5_vm.png", dpi=300)
plt.close()