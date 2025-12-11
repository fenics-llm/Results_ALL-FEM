from dolfin import *

# Geometry and mesh
L = 1.0; H = 0.10
mesh = RectangleMesh(Point(0,0), Point(L,H), 100, 10)

# Function space
V = FunctionSpace(mesh, "CG", 2)
c, v = TrialFunction(V), TestFunction(V)

# Boundary markers
class Inlet(SubDomain):
    def inside(self, x, on_boundary): return near(x[0], 0.0) and on_boundary

class Outlet(SubDomain):
    def inside(self, x, on_boundary): return near(x[0], L) and on_boundary

class Walls(SubDomain):
    def inside(self, x, on_boundary): return (near(x[1], 0.0) or near(x[1], H)) and on_boundary

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)
Inlet().mark(boundaries, 1); Outlet().mark(boundaries, 2); Walls().mark(boundaries, 3)

# Boundary conditions
bc_in = DirichletBC(V, Constant(0.0), boundaries, 1)
bc_out = DirichletBC(V, Constant(1.0), boundaries, 2)

# Velocity field and diffusivity
U_max = 0.75; D = 1e-5

u_x = Expression("4*U_max*x[1]*(H-x[1])/pow(H,2)", U_max=U_max, H=H, degree=2)
u = as_vector([u_x, Constant(0)])

# Variational formulation
h = CellDiameter(mesh); h_avg = (h('+') + h('-'))/2.0

n = FacetNormal(mesh)

a = inner(u, grad(c))*v*dx - D*inner(grad(c), grad(v))*dx \
    + 0.5*dot(u('+'), n('+'))*(c('+')-c('-'))*(v('+')-v('-'))*dS
L = Constant(0)*v*dx

# Solve the problem
c_sol = Function(V)
solve(lhs(a) == rhs(L), c_sol, [bc_in, bc_out])

# Output
file = XDMFFile("channel_c.xdmf")
file.parameters["flush_output"] = True
file.write(c_sol)