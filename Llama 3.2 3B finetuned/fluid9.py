from dolfin import *
import mshr

# Mesh and function space
mesh = mshr.Rectangle(Point(0, 0), Point(2, 0.20))
V = VectorFunctionSpace(mesh, "P", 1)

# Mark boundaries
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)
tol = 1e-6

class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, tol)

class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 2.0, tol)

class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], 0.0, tol) or near(x[1], 0.20, tol))

Inlet().mark(boundaries, 1)
Outlet().mark(boundaries, 2)
Walls().mark(boundaries, 3)

ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

# Flow (u, p)
u_x_inlet = Expression(("6.0*U_bar*(1.0-y[1]/H)", "0.0"), U_bar=0.1, H=0.20, degree=2)
u = TrialFunction(V); v = TestFunction(V)
f = Constant((0.0, 0.0))
a_flow = u_x_inlet[0]*v*dx + u[1]*div(v)*dx
L_flow = f[0]*v*dx

# Concentration (c)
kappa = Constant(1e-3) # m²/s
u_adv = project(u, V); c_inlet = Expression("x[1]", degree=2)

a_conc = kappa*dot(grad(c), grad(v))*dx + u_adv[0]*v*ds(1)
L_conc = Constant(0.0)*v*ds(1) # no flux on walls

# Solve
u, p, c = solve_system(a_flow + a_conc, L_flow + L_conc,
                      [Constant((0.0, 0.0)), Constant(0.0), c_inlet])