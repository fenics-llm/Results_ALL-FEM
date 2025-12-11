# filename: q2_stokes.py
from dolfin import *
import matplotlib.pyplot as plt

# Parameters
L = 2.0
H = 0.20
mu = 1.0
rho = 1.0
f = Constant((1.0, 0.0))

# Mesh
mesh = RectangleMesh(Point(0.0, 0.0), Point(L, H), 120, 12)

# Taylor–Hood spaces
V = VectorFunctionSpace(mesh, "Lagrange", 2)   # velocity
Q = FunctionSpace(mesh, "Lagrange", 1)         # pressure
Ve = V.ufl_element()
Pe = Q.ufl_element()
TH = MixedElement([Ve, Pe])
W = FunctionSpace(mesh, TH)                    # mixed (u,p)


# Trial/Test functions
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

# Walls (no-slip)
class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], 0.0) or near(x[1], H))
walls = Walls()
bcu = DirichletBC(W.sub(0), Constant((0.0, 0.0)), walls)

# Variational form
def eps(w):
    return sym(grad(w))
a = (2*mu*inner(eps(u), eps(v)) - div(v)*p + q*div(u))*dx
Lform = inner(f, v)*dx

# Assemble system
A, b = assemble_system(a, Lform, bcu)

# Pressure nullspace (constant)
null_vec = Vector()
A.init_vector(null_vec, 1)
null_vec.zero()
for d in W.sub(1).dofmap().dofs():
    null_vec[d] = 1.0
nullspace = VectorSpaceBasis([null_vec])
as_backend_type(A).set_nullspace(nullspace)
nullspace.orthogonalize(b)

# Solve
w = Function(W)
solve(A, w.vector(), b, "mumps")
(u_sol, p_sol) = w.split()

# Speed magnitude
speed = sqrt(dot(u_sol, u_sol))
V0 = FunctionSpace(mesh, "Lagrange", 2)
speed_proj = project(speed, V0)

# Plot speed
plt.figure(figsize=(8, 2))
c = plot(speed_proj, title="Speed |u|", cmap="viridis")
plt.colorbar(c)
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.savefig("q2_speed.png", dpi=300)

# Save solution
xdmf = XDMFFile(mesh.mpi_comm(), "q2_solution.xdmf")
xdmf.write(u_sol, 0.0)
xdmf.write(p_sol, 0.0)
xdmf.close()