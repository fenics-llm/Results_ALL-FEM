from dolfin import *

# Geometry and mesh
mesh = UnitSquareMesh(96, 96)

# Function spaces: Taylor–Hood P2–P1
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
ME = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, ME)

# Boundary markers
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0) # default value

class Left(SubDomain):
    def inside(self, x, on_boundary): return near(x[0], 0.0) and on_boundary
class Right(SubDomain):
    def inside(self, x, on_boundary): return near(x[0], 1.0) and on_boundary
class Bottom(SubDomain):
    def inside(self, x, on_boundary): return near(x[1], 0.0) and on_boundary
class Top(SubDomain):
    def inside(self, x, on_boundary): return near(x[1], 1.0) and on_boundary
class P0(SubDomain):
    def inside(self, x, on_boundary): return near(x[0], 0.0) and near(x[1], 0.0)

Left().mark(boundaries, 1)
Right().mark(boundaries, 2)
Bottom().mark(boundaries, 3)
Top().mark(boundaries, 4)

ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

# Trial/test functions
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

# Parameters
rho = Constant(1.0)
mu = Constant(1.0)

# Variational form
F = ( 2*mu*inner(sym(grad(u)), sym(grad(v))) - div(v)*p + q*div(u) )*dx

# Boundary conditions
noslip = Constant((0.0, 0.0))
lid   = Constant((1.0, 0.0))

bc_left    = DirichletBC(W.sub(0), noslip, boundaries, 1)
bc_right   = DirichletBC(W.sub(0), noslip, boundaries, 2)
bc_bottom  = DirichletBC(W.sub(0), noslip, boundaries, 3)
bc_top     = DirichletBC(W.sub(0), lid,    boundaries, 4)
bc_p0      = DirichletBC(W.sub(1), Constant(0.0), P0(), method="pointwise")

bcs = [bc_left, bc_right, bc_bottom, bc_top, bc_p0]

# Solve
w = Function(W)
solve(lhs(F) == rhs(F), w, bcs)

u_soln, p_soln = w.split()

# Postprocessing: speed |u|
speed = project(sqrt(dot(u_soln, u_soln)), Q,
                form_compiler_parameters={"representation":"quadrature"})

# Visualization
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
c = plot(speed, title="Speed |u|")
plt.colorbar(c)
plt.savefig("q3_speed.png")

# Save to XDMF
file = XDMFFile("q3_soln.xdmf")
file.parameters["flush_output"] = True
file.parameters["functions_share_mesh"] = True

file.write(u_soln, 0.0)
file.write(p_soln, 0.0)