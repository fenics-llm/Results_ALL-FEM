from dolfin import *
import mshr
import matplotlib.pyplot as plt

# Geometry
L = 1.0; H = 0.10
mesh = mshr.RectangleMesh(mshr.Rectangle(Point(0,0), Point(L,H)), 100, 10)
V = FunctionSpace(mesh,"P",1)

# Mark boundaries
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
eps=1e-6
class InBdry(SubDomain):
    def inside(self,x,on_b): return on_b and near(x[0],0.0,eps)
class OutBdry(SubDomain):
    def inside(self,x,on_b): return on_b and near(x[0],L+eps,eps)
class WallBdry(SubDomain):
    def inside(self,x,on_b): return on_b and (near(x[1],0.0,eps) or near(x[1],H+eps,eps))
InBdry().mark(boundaries,1); OutBdry().mark(boundaries,2); WallBdry().mark(boundaries,3)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

# Velocity
u_max = 0.75; Ux = Expression(("U_max*x[1]*(H-x[1])","0.0"),U_max=u_max,H=H,L=L,x=[0,0])
vel = interpolate(u_max*grad(Ux), VectorFunctionSpace(mesh,"P",1))

# Diffusion
D = Constant(1e-5)

# Variational formulation
c = TrialFunction(V)
v = TestFunction(V)
a = inner(D*grad(c), grad(v))*dx + vel[0]*c*v*dx

bc_inlet = DirichletBC(V, Constant(0.0), boundaries, 1)
bc_outlet = DirichletBC(V, Constant(1.0), boundaries, 2)

# Solve
solve(a == 0, c, [bc_inlet,bc_outlet])

# Output
file = File("concentration.xdmf")
file.parameters["flush_output"] = True
file << c

plt.figure()
p = plot(c)
plt.colorbar(p)
plt.title("Concentration (u_max=0.75 m/s)")
plt.show()
