# filename: q1_stokes.py
from dolfin import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# -------------------------------------------------
# 1. Geometry and mesh
# -------------------------------------------------
L, H = 2.0, 0.20
nx, ny = 100, 10
mesh = RectangleMesh(Point(0.0, 0.0), Point(L, H), nx, ny, "crossed")

# -------------------------------------------------
# 2. Boundary markers (1 = inlet, 2 = outlet)
# -------------------------------------------------
inlet_id, outlet_id = 1, 2
facets = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)

class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0)

class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], L)

Inlet().mark(facets, inlet_id)
Outlet().mark(facets, outlet_id)

ds = Measure("ds", domain=mesh, subdomain_data=facets)
n = FacetNormal(mesh)

# -------------------------------------------------
# 3. Physical parameters
# -------------------------------------------------
mu = Constant(1.0)      # Pa·s
p_in = Constant(1.0)    # Pa
p_out = Constant(0.0)   # Pa

# -------------------------------------------------
# 4. Mixed (Taylor–Hood) function space
# -------------------------------------------------
Ve = VectorElement("Lagrange", mesh.ufl_cell(), 2)   # velocity P2
Pe = FiniteElement("Lagrange", mesh.ufl_cell(), 1)   # pressure P1
W = FunctionSpace(mesh, MixedElement([Ve, Pe]))

# -------------------------------------------------
# 5. Boundary conditions
# -------------------------------------------------
def walls(x, on_boundary):
    return on_boundary and (near(x[1], 0.0) or near(x[1], H))

bc_walls = DirichletBC(W.sub(0), Constant((0.0, 0.0)), walls)

# Pin pressure at a single interior point to eliminate the nullspace
class PressurePoint(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and near(x[1], 0.0)

bc_p = DirichletBC(W.sub(1), Constant(0.0), PressurePoint(), "pointwise")

bcs = [bc_walls, bc_p]

# -------------------------------------------------
# 6. Variational formulation (steady Stokes)
# -------------------------------------------------
(wu, wp) = TrialFunctions(W)
(vu, vp) = TestFunctions(W)

def eps(u):
    return sym(grad(u))

a = (2.0*mu*inner(eps(wu), eps(vu)) - div(vu)*wp + vp*div(wu)) * dx
L_form = inner(-p_in*n, vu)*ds(inlet_id) + inner(-p_out*n, vu)*ds(outlet_id)

# -------------------------------------------------
# 7. Assemble and solve
# -------------------------------------------------
A, b = assemble_system(a, L_form, bcs)

w = Function(W, name="w")
solve(A, w.vector(), b, "mumps")
u, p = w.split(deepcopy=True)
u.rename("u", "velocity")
p.rename("p", "pressure")

# -------------------------------------------------
# 8. Save solution (XDMF)
# -------------------------------------------------
with XDMFFile(mesh.mpi_comm(), "q1_soln.xdmf") as xdmf:
    xdmf.parameters["flush_output"] = True
    xdmf.parameters["functions_share_mesh"] = True
    xdmf.write(u, 0.0)
    xdmf.write(p, 0.0)

# -------------------------------------------------
# 9. Speed magnitude and PNG plot
# -------------------------------------------------
Vsc = FunctionSpace(mesh, "CG", 1)
speed = project(sqrt(dot(u, u)), Vsc)
speed.rename("speed", "speed")

# Use Agg backend for headless environments
matplotlib.use("Agg")
coords = mesh.coordinates()
cells = mesh.cells()
tri = mtri.Triangulation(coords[:, 0], coords[:, 1], cells)
z = speed.compute_vertex_values(mesh)

plt.figure(figsize=(8, 2))
plt.tripcolor(tri, z, shading="gouraud")
plt.colorbar(label="|u| (m/s)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Speed magnitude |u|")
plt.tight_layout()
plt.savefig("q1_speed.png", dpi=300)
plt.close()

print("Simulation completed. Files written: q1_soln.xdmf, q1_speed.png")