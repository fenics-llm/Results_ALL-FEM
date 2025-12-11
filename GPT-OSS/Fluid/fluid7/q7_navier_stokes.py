# filename: q7_navier_stokes.py
from dolfin import *
import mshr
import matplotlib.pyplot as plt

# -------------------- Parameters --------------------
L = 2.2          # length (m)
H = 0.41         # height (m)
R = 0.05         # radius of hole (m)
center = Point(0.20, 0.20)

mu = 1e-3        # dynamic viscosity (Pa·s)
rho = 1.0        # density (kg/m³)
Ubar = 0.2       # mean inlet velocity (m/s)
D = 2.0 * R      # cylinder diameter

# -------------------- Mesh --------------------
domain = mshr.Rectangle(Point(0.0, 0.0), Point(L, H)) - mshr.Circle(center, R, 64)
mesh = mshr.generate_mesh(domain, 64)

# -------------------- Function Spaces (Taylor–Hood) --------------------
V = VectorFunctionSpace(mesh, "Lagrange", 2)   # velocity
Q = FunctionSpace(mesh, "Lagrange", 1)       # pressure

# Mixed space using a MixedElement (compatible with recent dolfin)
mixed_elem = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, mixed_elem)

# -------------------- Boundary Markers --------------------
class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and on_boundary

class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], L) and on_boundary

class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[1], 0.0) or near(x[1], H)) and on_boundary

class Cylinder(SubDomain):
    def inside(self, x, on_boundary):
        # points on the circular boundary
        return on_boundary and ((x[0]-center.x())**2 + (x[1]-center.y())**2 < (R+1e-8)**2)

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
Inlet().mark(boundaries, 1)
Outlet().mark(boundaries, 2)
Walls().mark(boundaries, 3)
Cylinder().mark(boundaries, 4)

ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# -------------------- Inlet Velocity Profile --------------------
class InletVelocity(UserExpression):
    def eval(self, values, x):
        y = x[1]
        values[0] = 6.0 * Ubar * y * (H - y) / H**2
        values[1] = 0.0
    def value_shape(self):
        return (2,)

u_inlet = InletVelocity(degree=2)

# -------------------- Variational Formulation --------------------
# Split mixed unknown
w = Function(W)               # (u, p) will be stored here
(u, p) = split(w)             # u: velocity, p: pressure
(v, q) = TestFunctions(W)     # test functions

def sigma(u, p):
    return 2.0 * mu * sym(grad(u)) - p * Identity(len(u))

# Residual (steady Navier–Stokes)
F = ( rho * dot(dot(u, nabla_grad(u)), v)
      + inner(sigma(u, p), grad(v))
      - div(v) * p
      + q * div(u) ) * dx

# -------------------- Boundary Conditions --------------------
bcu_inlet = DirichletBC(W.sub(0), u_inlet, boundaries, 1)               # inlet velocity
bcu_walls = DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 3) # top/bottom walls
bcu_cyl   = DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 4) # cylinder no‑slip

# Pin pressure at outlet (set p = 0 on the outlet boundary)
bcp_outlet = DirichletBC(W.sub(1), Constant(0.0), boundaries, 2)

bcs = [bcu_inlet, bcu_walls, bcu_cyl, bcp_outlet]

# -------------------- Nonlinear Solver --------------------
# Jacobian of F
J = derivative(F, w, TrialFunction(W))

problem = NonlinearVariationalProblem(F, w, bcs, J)
solver = NonlinearVariationalSolver(problem)

prm = solver.parameters['newton_solver']
prm['relative_tolerance'] = 1e-6
prm['absolute_tolerance'] = 1e-8
prm['maximum_iterations'] = 25
prm['linear_solver'] = 'mumps'

solver.solve()

# Extract velocity and pressure
(u_sol, p_sol) = w.split()

# -------------------- Drag Coefficient --------------------
n = FacetNormal(mesh)
stress = sigma(u_sol, p_sol)

# Drag = ∫_cylinder (σ·n)·e_x ds
drag = assemble(dot(dot(stress, n), Constant((1.0, 0.0))) * ds(4))
C_D = 2.0 * drag / (rho * Ubar**2 * D)

print("Drag force (N) :", drag)
print("Drag coefficient C_D :", C_D)

# -------------------- Speed Plot --------------------
speed = sqrt(dot(u_sol, u_sol))
V0 = FunctionSpace(mesh, "CG", 1)
speed_proj = project(speed, V0)

plt.figure(figsize=(8, 3))
p = plot(speed_proj, title="Speed magnitude |u| (m/s)", cmap='viridis')
plt.colorbar(p)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.tight_layout()
plt.savefig('q7_speed.png', dpi=300)
plt.close()

# -------------------- Save Solution --------------------
xdmf = XDMFFile(mesh.mpi_comm(), "q7_soln.xdmf")
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True
xdmf.write(u_sol, 0.0)
xdmf.write(p_sol, 0.0)
xdmf.close()