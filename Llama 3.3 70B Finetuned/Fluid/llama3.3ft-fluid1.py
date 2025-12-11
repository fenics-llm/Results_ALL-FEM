from dolfin import *
import mshr

# Geometry
L = 2.0; H = 0.20
domain = mshr.Rectangle(Point(0, 0), Point(L, H))
mesh = mshr.generate_mesh(domain, 100)

# Function space: P2 for velocity and P1 for pressure
V = VectorFunctionSpace(mesh, "CG", 2); Q = FunctionSpace(mesh, "CG", 1)
ME = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, ME)

# Boundary markers
class Inlet(SubDomain):
    def inside(self, x, on_boundary): return near(x[0], 0.0) and on_boundary

class Outlet(SubDomain):
    def inside(self, x, on_boundary): return near(x[0], L) and on_boundary

class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[1], 0.0) or near(x[1], H)) and on_boundary

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0); Inlet().mark(boundaries, 1); Outlet().mark(boundaries, 2); Walls().mark(boundaries, 3)

ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

# Model parameters
mu = Constant(1.0)
rho = Constant(1.0)

# Boundary conditions: no-slip on walls (u=0), traction-free outlet
noslip = Constant((0.0, 0.0))
bc_walls = DirichletBC(W.sub(0).collapse(), noslip, boundaries, 3)

p_in = Constant(1.0)
p_out = Constant(0.0)

# Variational formulation
(u, p) = TrialFunctions(W); (v, q) = TestFunctions(W)
f = Constant((0.0, -rho*9.81))
a = mu * inner(grad(u), grad(v)) * dx - p * div(v) * dx + q * div(u) * dx
L = dot(f, v) * dx

# Traction terms on inlet and outlet
n = FacetNormal(mesh)
a += -p_in * dot(n, v) * ds(1)
L += -p_out * dot(n, v) * ds(2)

# Solve the system
w = Function(W); solve(a == L, w, bc_walls)

u_sol, p_sol = w.split()

# Postprocessing: speed |u|
V0 = u_sol.function_space()
speed = project(sqrt(dot(u_sol, u_sol)), V0)
plot(speed, title="Speed |u|"); interactive()

# Save results
File("q1_speed.png") << speed
File("q1_soln.xdmf") << u_sol; File("q1_soln.xdmf") << p_sol