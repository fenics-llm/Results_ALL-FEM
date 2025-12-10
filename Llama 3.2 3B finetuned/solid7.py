from dolfin import *
import mshr

# Geometry
mesh = mshr.Rectangle(Point(0, 0), Point(1, 0.20)) + \
       mshr.Rectangle(Point(0, 0.10), Point(1, 0.20))
mesh = mshr.generate_mesh(mesh, 80)

# Material subdomains
class Al(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[1], 0.20) and on_boundary)
class Steel(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[1], 0.10) and on_boundary)

# Mesh function
al = Al()
steel = Steel()

# Function space
V = VectorFunctionSpace(mesh, "P", 1)

# Mark material interfaces
facets = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
facets.set_all(0)
al.mark(facets, 1); steel.mark(facets, 2)
ds = Measure("ds", domain=mesh, subdomain_data=facets)

# Variational problem
E_al, nu_al = 70e9, 0.3; E_steel, nu_steel = 200e9, 0.3
mu_al, lambda_al = E_al / (2 * (1 + nu_al)), E_al * nu_al / ((1 + nu_al) * (1 - 2 * nu_al))
mu_steel, lambda_steel = E_steel / (2 * (1 + nu_steel)), E_steel * nu_steel / ((1 + nu_steel) * (1 - 2 * nu_steel))

u, v = TrialFunction(V), TestFunction(V)
f = Constant((0.0, -5000.0))
a = mu_al*inner(grad(u), grad(v))*dx \
    + lambda_al*div(v)*div(u)*dx \
    + E_steel/(1+nu_steel)*(dot(n, u)-u[1]*n[1])*dot(n, v)-\
    E_steel/(1+nu_steel)*(dot(n, v)-v[1]*n[1])*dot(n, u)

L = f*v*dx

# Boundary conditions
bc_left = DirichletBC(V, Constant((0.0, 0.0)), "near(x[0], 0.0)")

# Solve
u_sol = Function(V)
solve(a == L, u_sol, bc_left)

# Output
import matplotlib.pyplot as plt
plt.figure(); c=plot(u_sol); plt.colorbar(c)
plt.savefig("q7_disp.png")