from dolfin import *
import mshr

# 1. Create mesh and function space
mesh = mshr.Rectangle(Point(0, 0), Point(1, 1)).subdivide(64)
V = VectorFunctionSpace(mesh, "P", 2)

# 2. Define boundary markers
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)

class PeriodicBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0) or near(x[0], 1.0)

PeriodicBoundary().mark(boundaries, 1)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

# 3. Define velocity and pressure
u, v = TrialFunctions(V), TestFunctions(V)
p = Function(V)

# 4. Define initial conditions
u0 = Expression(("sin(2*pi*x[0])*cos(2*pi*x[1])", "-cos(2*pi*x[0])*sin(2*pi*x[1])"), degree=3)
t = 0

# 5. Define time-stepping parameters
dt = 0.25
T = 1.0
n_steps = int(T/dt)

# 6. Define function space for velocity and pressure at each time step
V_t = VectorFunctionSpace(mesh, "P", 2)
p_t = FunctionSpace(mesh, "P", 1)

# 7. Define the unsteady Navier–Stokes equations
rho, nu = Constant(1.0), Constant(1e-3)
f = Constant((0.0, -9.81))
g = Constant((0.0, 0.0))

def u_dot(u):
    return dot(grad(u), grad(u))

# 8. Define the time derivative of velocity
u_t = Function(V)

for n in range(1, n_steps+1):

    # 9. Update time and mesh
    t += dt

    # 10. Compute velocity at current time step
    u_n = interpolate(u0, V)
    u_n.assign(project(u(t), V))

    # 11. Define the unsteady Navier–Stokes equations
    a = rho*dot(u_t, v)*dx + nu*u_dot(v)*dx
    L = rho*f*v*dx + dot(nu*grad(u), v)*ds(1) - p*t*v*dx

    # 12. Solve for pressure at current time step
    solve(a == L, p_t)

    # 13. Compute the velocity at next time step
    u_n.assign(project(u(t+dt), V))

# 14. Save velocity field in XDMF format
File("velocity.pvd") << u