from dolfin import *
import mshr

# Geometry
L = 2.0; H = 0.20
mesh = RectangleMesh(Point(0,0), Point(L,H), 100,10)
V = VectorFunctionSpace(mesh,"P",1)

# Boundary markers
bdr = MeshFunction("size_t", mesh, mesh.topology().dim()-1); bdr.set_all(0)
tol=1e-6
class Inlet(SubDomain):
    def inside(self,x,on_b): return on_b and near(x[0],0,tol)
class Outlet(SubDomain):
    def inside(self,x,on_b): return on_b and near(x[0],L,tol)
Inlet().mark(bdr,1); Outlet().mark(bdr,2)
ds = Measure("ds", domain=mesh, subdomain_data=bdr)

# Variational formulation
mu = Constant(1.0); rho = Constant(1.0)
u,v = TrialFunction(V), TestFunction(V)
p,q = TrialFunction(V), TestFunction(V)
f = Constant((0.0, -9.81))
a = mu*inner(grad(u),grad(v))*dx + div(v)*p*dx + q*div(u)*dx
L = dot(f,v)*dx

# Boundary conditions
bc_in = DirichletBC(V.sub(0).sub(0), Constant((1.0,0.0)), bdr, 1)
bc_out = DirichletBC(V.sub(0).sub(0), Constant((0.0,0.0)), bdr, 2)

# Solve
u_sol,p_sol = Function(V), Function(V)
solve(a == L, u_sol, bc_in + bc_out, solver_parameters={"linear_solver":"mumps"})
p = project(-rho*dot(grad(u_sol),grad(q))*dx, V)

# Postprocessing
speed = project(sqrt(dot(u_sol,u_sol)), V); speed.rename("Speed", None)
plt.figure(); p.plot(); plt.colorbar(); plt.title("Pressure (Pa)"); plt.savefig("q1_pressure.png")
plt.figure(); speed.plot(); plt.colorbar(); plt.title("|u| (m/s)"); plt.savefig("q1_speed.png")
