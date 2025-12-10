from dolfin import *
import mshr

# Geometry
L = 2.0; H = 0.20
mesh = mshr.Rectangle(Point(0,0), Point(L,H))*mshr.Rectangle(Point(0,0), Point(L,1))
mesh = mesh.subdivide(120,12)

# Function space and boundary markers
V = VectorFunctionSpace(mesh,"P",1)
bmark_in = MeshFunction("size_t",mesh,mesh.topology().dim()-1); bmark_in.set_all(0)
class In(SubDomain):
    def inside(self,x,on): return on and near(x[0],0.0,1e-10)
In().mark(bmark_in,1)  # inlet
class Out(SubDomain):
    def inside(self,x,on): return on and near(x[0],L+1e-10)
Out().mark(bmark_in,2)  # outlet

# Body force f = (1.0, 0.0) N/m³
f = Constant((1.0,0.0))

# Velocity: u = (u_x,u_y), pressure p
u, v = TrialFunction(V), TestFunction(V)
p, q = TrialFunction(V), TestFunction(V)

# Variational form
a = inner(grad(u), grad(v))*dx + div(v)*p*dx - div(u)*q*dx
L = dot(f,v)*dx

# No-slip on walls (Γ_w)
bc_w = DirichletBC(V.sub(0).sub(1), Constant((0.0,0.0)), bmark_in, 3)

# Inlet/outlet: traction-free natural condition (−p I + μ(∇u + ∇uᵀ)) n = 0
bc_inout = DirichletBC(V.sub(0).sub(1), Constant((0.0,0.0)), bmark_in, 1|2)

# Solve
uh = Function(V); solve(a == L, uh, [bc_w,bc_inout])
p_h = project(-div(uh), V)
u_h, p_h

# Save
import matplotlib.pyplot as plt
plt.figure(); c=plot(|uh|,(0.0,1.0)); plt.colorbar(c); plt.savefig("q2_speed.png")
plt.close()
File("q2_solution.xdmf") << uh,p_h
