from dolfin import *
from mshr import Rectangle, generate_mesh
import matplotlib.pyplot as plt

# Geometry
mesh = generate_mesh(Rectangle(Point(0, 0), Point(2.0, 0.2)), 4000)
V_elem = VectorElement("CG", mesh.ufl_cell(), 2)
Q_elem = FiniteElement("CG", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, MixedElement([V_elem, Q_elem]))
ME = FunctionSpace(mesh, MixedElement([V_elem, Q_elem]))

# Velocity inlet profile
U_bar = 0.1
H = 0.2
u_in = Expression(("6*U_bar*(y[1]/H)*(1 - y[1]/H)", "0.0"),
                  U_bar=U_bar, H=H, degree=2)

# Boundary markers
bndry = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
bndry.set_all(0)
class Inlet(SubDomain):
    def inside(self, x, on): return on and near(x[0], 0.0)
class Walls(SubDomain):
    def inside(self, x, on): return on and (near(x[1], 0.0) or near(x[1], 0.2))
class Outlet(SubDomain):
    def inside(self, x, on): return on and near(x[0], 2.0)
Inlet().mark(bndry, 1)
Walls().mark(bndry, 2)
Outlet().mark(bndry, 3)
ds = Measure("ds", domain=mesh, subdomain_data=bndry)

# Navier-Stokes
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
rho, mu = 1.0, 0.01
f = Constant((0.0, 0.0))
a = (rho*dot(u, v)*dx + rho*dot(dot(u, nabla_grad(u)), v)*dx
     + mu*inner(grad(u), grad(v))*dx - div(v)*p*dx - q*div(u)*dx)
L = -rho*dot(u_in, v)*ds(1)

bc_inlet = DirichletBC(W.sub(0), interpolate(u_in, V_elem), bndry, 1)
bc_walls = DirichletBC(W.sub(0), Constant((0.0, 0.0)), bndry, 2)
bcs = [bc_inlet, bc_walls]

w = Function(W)
solve(a == L, w, bcs, solver_parameters={"linear_solver": "mumps"})
u_sol, p_sol = w.split(deepcopy=True)

# Advection-diffusion
Vc = FunctionSpace(mesh, "CG", 1)
c, c_test = TrialFunction(Vc), TestFunction(Vc)
kappa = 1e-3
a_adv = kappa*dot(grad(c), grad(c_test))*dx + dot(u_sol, grad(c))*c_test*dx
L_adv = Constant(1.0)*c_test*ds(3)
bc_inlet_c = DirichletBC(Vc, Constant(0.0), bndry, 1)
c_sol = Function(Vc)
solve(a_adv == L_adv, c_sol, bc_inlet_c)

# Save outputs
plt.figure(); c_plot = plot(c_sol); plt.colorbar(c_plot); plt.savefig("q10_conc.png")
with XDMFFile("q10_solution.xdmf") as file:
    file.write(u_sol, 0.0)
    file.write(p_sol, 0.0)
    file.write(c_sol, 0.0)
