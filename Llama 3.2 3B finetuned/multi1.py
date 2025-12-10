from dolfin import *
import mshr

# Mesh and function space
mesh = mshr.Circle(Point(0,0), 0.05) + mshr.Rectangle(Point(-2e-3,-2e-3), Point(2e-3,2e-3))
domain = mesh
for t in range(100):
    w = Expression("s*x[0]/pow(x[0]*x[0]+x[1]*x[1],0.5)", s=1e-4, degree=2)
    mesh = mshr.movepoint(mesh,w)
    domain = domain + mesh

mesh = mshr.generate_mesh(domain, 64)

V = FunctionSpace(mesh,"P",1)
c = TrialFunction(V)
w = TestFunction(V)

# Boundary markers
bdr = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
class Inlet(SubDomain):
    def inside(self,x,on_b): return on_b and near(x[0],0.0)
class Outlet(SubDomain):
    def inside(self,x,on_b): return on_b and near(x[0],2.0)
Inlet().mark(bdr,1); Outlet().mark(bdr,2)
ds = Measure("ds", domain=mesh, subdomain_data=bdr)

# Variational form
a = D*dot(grad(c),grad(w))*dx + kappa*c*w*dx

# BCs
bc = DirichletBC(V, Constant(0.0), bdr, 1) # inlet
bcs = [bc]

# Time loop
c_old = Function(V)
for t in range(n_steps):
    c_new = Function(V)
    solve(a == w*c_new, c_new, bcs)
    c_old.assign(c_new)

    if (t+1)%100==0:
        print("Time step %d: L² norm of concentration = %.3e" %
              (t+1, assemble(pow(c_new(0),2)*dx)))

# Save output
File("disk_c.pvd") << c