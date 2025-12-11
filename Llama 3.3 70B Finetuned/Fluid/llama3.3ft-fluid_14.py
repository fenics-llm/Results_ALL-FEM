from dolfin import *
import mshr

# Geometry
D = 1.0
domain = mshr.Rectangle(Point(-30*D, -30*D), Point(30*D, 30*D)) \
       - mshr.Circle(Point(0, 0), D/2)
mesh = mshr.generate_mesh(domain, 180)

# Function space
V = VectorFunctionSpace(mesh, "CG", 2); Q = FunctionSpace(mesh, "CG", 1)
ME = MixedElement([V.ufl_element(), Q.ufl_element()]); W = FunctionSpace(mesh, ME)

# Boundary markers
class Inflow(SubDomain):
    def inside(self, x, on_boundary): return near(x[0], -30*D) and on_boundary

class Outflow(SubDomain):
    def inside(self, x, on_boundary): return near(x[0], 30*D) and on_boundary

class Cylinder(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (x[0]**2 + x[1]**2 <= D**2/4.0)

class TopBottom(SubDomain):
    def inside(self, x, on_boundary): return near(abs(x[1]), 30*D) and on_boundary

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0); Inflow().mark(boundaries, 1); Outflow().mark(boundaries, 2)
Cylinder().mark(boundaries, 3); TopBottom().mark(boundaries, 4)

ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

# Parameters
U = 1.0; mu = Constant(2.56e-5); rho = Constant(1.0)
dt = Constant(0.01); t_end = 10.0

# Variational multiscale formulation
u, p = TrialFunctions(W); v, q = TestFunctions(W)

w_old = Function(W) # initialize w_old before time loop
u_old, p_old = split(w_old)

a = (rho*inner(u - u_old, v) + dt*(mu*inner(grad(u), grad(v)) +
    rho*inner(dot(grad(u), u_old), v))) * dx

L = rho*dot(u_old, v)*dx
A, b = assemble_system(a, L)

# Stabilization terms
h = CellDiameter(mesh)
tau_M = (4.0/(dt*rho) + 9.0*(mu/h**2) +
    (rho*u_old[0]**2)/h)**(-1/2)
tau_C = h*h/(12.0*mu)

a += tau_M * inner(dot(grad(u), u_old), dot(grad(v), u_old)) * dx
L += tau_M * rho * dot(Constant((0, 0)), v) * dx

a += dt * tau_C * div(u) * div(v) * dx
L += dt * tau_C * Constant(0.0) * div(v) * dx

# Boundary conditions
u_in = Expression(("U", "0.0"), U=U)
noslip = Constant((0.0, 0.0))

bc1 = DirichletBC(W.sub(0), u_in, boundaries, 1)
bc2 = DirichletBC(W.sub(0).sub(1), noslip.sub(1), boundaries, 4)

bcs = [bc1, bc2]

# Time stepping
w = Function(W); t = 0.0

while t < t_end + DOLFIN_EPS:
    A, b = assemble_system(a, L, bcs)
    solve(A, w.vector(), b)

    u_h, p_h = w.split()
    w_old.assign(w) # update w_old for next time step
    u_old, p_old = split(w_old)

    t += dt(0.0)

# Postprocessing
drag = assemble(-p_h * ds(3))
print("Mean drag coefficient:", drag / (rho * U**2))

file = XDMFFile("vms_solution.xdmf")
file.parameters["flush_output"] = True
file.parameters["rewrite_function_mesh"] = False

u_h, p_h = w.split()
file.write(u_h, t)
file.write(p_h, t)
