from dolfin import *
from mshr   import Rectangle, generate_mesh

# Geometry and mesh
Lx, Ly = 2.0, 0.20
domain = Rectangle(Point(0, 0), Point(Lx, Ly))
mesh   = generate_mesh(domain, 200)

# Function spaces
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
ME = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, ME)

# Boundary markers
class Inlet(SubDomain):
    def inside(self, x, on_boundary): return near(x[0], 0.0) and on_boundary

class Outlet(SubDomain):
    def inside(self, x, on_boundary): return near(x[0], Lx) and on_boundary

class Walls(SubDomain):
    def inside(self, x, on_boundary): return (near(x[1], 0.0) or near(x[1], Ly)) and on_boundary

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
Inlet().mark(boundaries, 1)
Outlet().mark(boundaries, 2)
Walls().mark(boundaries, 3)

ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

# Parameters
rho = Constant(1.0)         # density (kg m⁻³)
mu = Constant(0.01)          # dynamic viscosity (Pa s)
kappa = Constant(1e-3)       # diffusivity (m² s⁻¹)

Ubar = 0.1                  # mean inflow speed
H = Ly

# Flow problem
u, p = TrialFunctions(W)
v, q = TestFunctions(W)

f = Constant((0.0, 0.0))     # body force per unit volume (N m⁻³)

a_flow = rho * inner(dot(u, nabla_grad(u)), v) * dx \
       + mu * inner(grad(u), grad(v)) * dx \
       - p * div(v) * dx \
       + q * div(u) * dx

L_flow = dot(f, v) * dx

# Boundary conditions
u_inlet = Expression(("6*Ubar*x[1]*(1-x[1]/H)", "0.0"), Ubar=Ubar, H=H, degree=2)
noslip = Constant((0.0, 0.0))

bc_flow_inlet = DirichletBC(W.sub(0), u_inlet, boundaries, 1)

bcs_flow = [bc_flow_inlet]

# Solve flow problem
w = Function(W)
solve(a_flow == L_flow, w, bcs_flow)

u_sol, p_sol = w.split()

# Advection-diffusion problem
c, v_c = TrialFunction(V), TestFunction(V)

a_advdiff = dot(u_sol, grad(c)) * v_c * dx \
          + kappa * inner(grad(c), grad(v_c)) * dx

L_advdiff = Constant(0.0) * v_c * dx

# Boundary conditions
c_inlet = Constant(0.0)
bc_advdiff_inlet = DirichletBC(V, c_inlet, boundaries, 1)

# Outlet condition: c(x=2)=1
bc_advdiff_outlet = DirichletBC(V, Constant(1.0), boundaries, 2)

bcs_advdiff = [bc_advdiff_inlet, bc_advdiff_outlet]

# Solve advection-diffusion problem
c_sol = Function(V)
solve(a_advdiff == L_advdiff, c_sol, bcs_advdiff)

# Save results
file_c = File("q10_conc.png")
file_c << c_sol

file_all = XDMFFile("q10_solution.xdmf")
file_all.parameters["flush_output"] = True
file_all.write(u_sol, 0.0)
file_all.write(p_sol, 0.0)
file_all.write(c_sol, 0.0)