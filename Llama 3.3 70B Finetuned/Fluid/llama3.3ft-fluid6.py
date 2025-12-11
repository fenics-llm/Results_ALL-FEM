from dolfin import *
import mshr as msh
import matplotlib.pyplot as plt

# Geometry parameters
H = 1.0
L_upstream = 3*H
L_downstream = 20*H

# Create the domain with a backward-facing step
domain = (msh.Rectangle(Point(-L_upstream, 0), Point(0, H)) +
          msh.Rectangle(Point(0, 0), Point(L_downstream, 2*H)))
mesh = msh.generate_mesh(domain, 180)

# Function space: P2 for velocity and P1 for pressure
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
ME = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, ME)

# Boundary markers
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0) # Initialize all boundaries as 0

class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], -L_upstream) and on_boundary
Inlet().mark(boundaries, 1)

class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], L_downstream) and on_boundary
Outlet().mark(boundaries, 2)

class BottomWall(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[1], 0.0) or
                (x[0] >= -DOLFIN_EPS and near(x[1], H))) and on_boundary
BottomWall().mark(boundaries, 3)

class TopWall(SubDomain):
    def inside(self, x, on_boundary):
        return ((near(x[1], H) and x[0] < DOLFIN_EPS) or
                (near(x[1], 2*H) and x[0] >= -DOLFIN_EPS)) and on_boundary
TopWall().mark(boundaries, 4)

class Step(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and (x[1] > H + DOLFIN_EPS) and on_boundary
Step().mark(boundaries, 5)

ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
n = FacetNormal(mesh)

# Parameters
rho = Constant(1.0)
mu = Constant(0.01)
U_bar = Constant(1.0)

# Inlet profile (parabolic)
u_inlet_expr = Expression(("6*U_bar*x[1]*(H-x[1])/(H*H)", "0.0"),
                          U_bar=U_bar, H=H, degree=2)

# Variational formulation
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
f = Constant((0.0, 0.0))

F = (rho*inner(dot(grad(u), u), v)*dx +
     mu*inner(grad(u), grad(v))*dx -
     div(v)*p*dx + q*div(u)*dx)

a = lhs(F)
L = rhs(F)

# Boundary conditions
bc_inlet = DirichletBC(W.sub(0), u_inlet_expr, boundaries, 1)
bcs = [bc_inlet,
       DirichletBC(W.sub(0).sub(0), Constant(0.0), boundaries, 3),
       DirichletBC(W.sub(0).sub(1), Constant(0.0), boundaries, 3),
       DirichletBC(W.sub(0).sub(0), Constant(0.0), boundaries, 4),
       DirichletBC(W.sub(0).sub(1), Constant(0.0), boundaries, 4),
       DirichletBC(W.sub(0).sub(0), Constant(0.0), boundaries, 5),
       DirichletBC(W.sub(0).sub(1), Constant(0.0), boundaries, 5)]

# Solve
w = Function(W)
solve(a == L, w, bcs)

u_h, p_h = w.split()

# Compute wall shear stress on the top wall (y=H or y=2H)
tau_w = project(mu*dot(grad(u_h)[1,0], n), FunctionSpace(mesh, "CG", 1),
                form_compiler_parameters={"representation":"quadrature"})

# Plot velocity
plt.figure()
plot(u_h, title="Velocity")
plt.savefig("q6_u.png")

# Save solution to XDMF file
file = XDMFFile("q6_soln.xdmf")
file.parameters["functions_share_mesh"] = True
file.parameters["rewrite_function_mesh"] = False
file.write(u_h, 0.0)
file.write(p_h, 0.0)
