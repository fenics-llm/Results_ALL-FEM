from dolfin import *
import mshr

# Geometry
L = 2.0; H = 0.20
mesh = mshr.Rectangle(Point(0,0), Point(L,H)) + \
       mshr.Rectangle(Point(0,0), Point(L,H/2))
mesh = mesh.subdivide(240,24)

# Material (Power-Law Fluid)
mu_0 = Constant(0.5); n = 0.5
def mu_eff(D):
    return mu_0*((D[0,0]**n + D[1,1]**n)**(1/n) + 10e-8)

# Function space and boundary markers
V = VectorFunctionSpace(mesh,"P",1)
bmarks = MeshFunction("size_t",mesh,mesh.topology().dim()-1,0)
bmarks.set_all(0)
class Inlet(SubDomain):
    def inside(self,x,on): return on and near(x[0],0.0,1e-3)
Inlet().mark(bmarks,1); class Outlet(SubDomain):
    def inside(self,x,on): return on and near(x[0],L+1e-3,1e-3)
Outlet().mark(bmarks,2)

# Variational form
u = TrialFunction(V); v = TestFunction(V)
rho = Constant(1.0)
f = Constant((0.0,0.0))
mu_eff_D = project(mu_eff(D(u)),V)
a = rho*dot(u,v)*dx + dot(mu_eff_D,D(v))*dx
L = f*v*dx

# Boundary conditions
bc_inlet = DirichletBC(V.sub(0), Expression("6*x[1]*(H-x[1])/(H^2)",H=H,U_bar=1.0),
                       bmarks,1)
bc_outlet = DirichletBC(V.sub(0), Constant(0.0), bmarks,2)

# Solve
u_sol = Function(V); solve(a == L, u_sol, [bc_inlet,bc_outlet])
p_sol = project(-dot(u_sol,D(u_sol)),V)
mu_eff_sol = project(mu_eff_D,u_sol)

# Output
import matplotlib.pyplot as plt
plt.figure(); c=plot(abs(u_sol),title="q12_speed.png"); plt.colorbar(c)
plt.figure(); plot(u_sol[0],y=mesh.coordinates()[1],"-o",label="ux(x,0.5)"); 
plt.figure(); plot(mu_eff_sol,x=mesh.coordinates()[0]); plt.title("mu_eff(x,y)")
plt.figure(); plot(p_sol,x=mesh.coordinates()[0]); plt.title("p(x,y)")
plt.savefig("q12_solution.png")

# Extract profile
y = mesh.coordinates()[:,1]
u_x = project(u_sol[0],V); u_y = project(u_sol[1],V)
with open("q12_profile.csv","w") as f:
    f.write("# x y ux\n")
    for i in range(mesh.num_vertices()):
        f.write("%f %f %f\n" % (mesh.coordinates()[i,0],
                                 y[i],u_x.project_value(i)))