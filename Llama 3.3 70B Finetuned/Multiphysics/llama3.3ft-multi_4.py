from dolfin import *

# Geometry
mesh = RectangleMesh(Point(0, -1), Point(pi, 1), 50, 20)

# Function spaces
V_S = VectorFunctionSpace(mesh, "CG", 2)
Q_S = FunctionSpace(mesh, "CG", 1)
ME_D = MixedElement([FiniteElement("DG", triangle, 0),
                     FiniteElement("BDM", triangle, 1)])
W_D = FunctionSpace(mesh, ME_D)

# Subdomains
class Stokes(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (x[1] >= -DOLFIN_EPS or near(x[0], 0) or near(x[0], pi))

class Darcy(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (x[1] <= DOLFIN_EPS + 1.0 or near(x[0], 0) or near(x[0], pi))

# Boundary markers
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)

stokes = Stokes()
darcy = Darcy()

ds_stokes = Measure("ds", domain=mesh, subdomain_data=boundaries)
ds_darcy = Measure("ds", domain=mesh, subdomain_data=boundaries)

# Parameters
g = 1.0
rho = 1.0
nu = 1.0
k = 1.0
K = k * rho * g / nu
alpha = 1.0

w_y = Expression("y", degree=2)
b_x = Expression("(nu*K - (alpha*g)/(2*nu))*x[1] - g/2.0*cos(x[0])",
                 nu=nu, K=K, alpha=alpha, g=g, degree=3)

b_y = Expression(
    "((nu*K)/2.0 - (alpha*g)/(4.0*nu))*pow(x[1], 2) - (g/2.0)*x[1] + ((alpha*g)/(2.0*nu) - 2.0*nu*K)*sin(x[0])",
    nu=nu, K=K, alpha=alpha, g=g, degree=3)

# Stokes boundary conditions
u_stokes = Expression(
    ("w_prime*cos(x[0])", "w*sin(x[0])"),
    w=lambda y: -K - (g * y) / (2.0 * nu) + (K / 2.0 - alpha * g /
                                            (4.0 * nu**2)) * y**2,
    w_prime=lambda y: (-g / (2.0 * nu)) +
                      (K / 2.0 - alpha * g / (4.0 * nu**2)) * 2.0 * y,
    K=K, alpha=alpha, nu=nu, g=g, degree=3)

# Darcy boundary conditions
p_darcy = Expression("rho*g*exp(x[1])*sin(x[0])", rho=rho, g=g, degree=3)

# Variational formulation
(u_S, p_S) = TrialFunctions(V_S * Q_S)
(v_S, q_S) = TestFunctions(V_S * Q_S)

a_stokes = (inner(grad(v_S), grad(u_S)) - div(v_S)*p_S + q_S*div(u_S))*dx

L_stokes = inner(b_x, v_S)*dx

# Darcy
(p_D, u_D) = TrialFunctions(W_D)
(q_D, v_D) = TestFunctions(W_D)

a_darcy = (inner(grad(p_D), grad(q_D)) + div(v_D)*p_D +
           q_D*div(u_D))*dx

L_darcy = Constant(0.0)*q_D*dx

# Interface conditions
n = Constant((0, -1))
t = Constant((1, 0))

a_int = (inner(dot(n, sigma(u_S, p_S)), v_S) +
         inner(t, dot(sigma(u_S, p_S), n)) * q_D +
         alpha / sqrt(k) * inner(dot(t, u_S - u_D), dot(t, v_S))) * ds(1)

L_int = (inner(p_D*n/float(rho), v_S) + alpha /
         float(sqrt(k)) * inner(dot(t, u_D), dot(t, v_S))) * ds(1)

# Assemble the system
a = a_stokes + a_darcy + a_int
L = L_stokes + L_darcy

# Boundary conditions
bc_u_stokes = DirichletBC(V_S, u_stokes, boundaries, 2)
bc_p_darcy = DirichletBC(Q_D, p_darcy, boundaries, 3)

bcs = [bc_u_stokes, bc_p_darcy]

# Solve the system
w = Function(W_S * W_D)
solve(a == L, w, bcs)

u_s, p_s = w.split()

File("velocity.pvd") << u_s
File("pressure.pvd") << p_s
