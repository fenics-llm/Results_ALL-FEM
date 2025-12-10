from dolfin import *
import mshr

# Geometry
L = 2.0; H = 0.20
mesh = mshr.Rectangle(Point(0,0), Point(L,H)) + \
       mshr.Rectangle(Point(0,0), Point(L,H))
mesh = mesh.subdivide(160,16)

# Function space
V = VectorFunctionSpace(mesh,"P",1)
u,v,T = TrialFunction(V), TestFunction(V), Function(V)

# Boundary markers
bmarks = MeshFunction("size_t",mesh,mesh.topology().dim()-1); bmarks.set_all(0)
class Inlet(SubDomain):
    def inside(self,x,on): return on and near(x[0],0.0,1e-10)
class Outlet(SubDomain):
    def inside(self,x,on): return on and near(x[0],2.0,1e-10)
class Walls(SubDomain):
    def inside(self,x,on): return on and (near(x[1],0.0,1e-10) or
                                       near(x[1],H,1e-10))
Inlet().mark(bmarks,1); Outlet().mark(bmarks,2); Walls().mark(bmarks,3)
ds = Measure("ds",domain=mesh,subdomain_data=bmarks)

# Inlet condition
u_in = Expression(("6*U*(x[1]/H - x[1]/H)*cos(pi*x[0]/L)",
                   "0.0"), U=2.5,H=H,L=L,precision=15)
bc_in = DirichletBC(V.sub(0), u_in, bmarks,1)

# No-slip walls
noslip = Constant((0.0, 0.0))

# Traction-free outlet
def outlet_traction(x,on):
    return on and near(x[0],2.0,1e-10)
bc_out = DirichletBC(V.sub(0), noslip, bmarks,3)

# Variational form
f = Constant((0.0, -9.81))
a = (mu*inner(grad(u), grad(v)) + div(T)*v*dx +
     u[1]*div(T)*v[0]*ds(2) -
     u[0]*div(T)*v[1]*ds(3))/dx
L = dot(f,v)*dx

# Solve
u_sol, p_sol = Function(V).split(), Function(V.sub(0))
solve(a == L, [u_sol,p_sol], [bc_in,bc_out])

# Save
plot(u_sol); plot(p_sol); plot(mesh); interactive()
