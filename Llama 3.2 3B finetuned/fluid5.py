from dolfin import *
import mshr

# Geometry
mesh = mshr.Rectangle(Point(0, 0), Point(1, 1)) - \
       mshr.Circle(Point(0.5, 0.5), 0.25)
V = VectorFunctionSpace(mesh, "P", 2)

# Boundary markers
bmarks = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
bmarks.set_all(0); eps = 1e-6

class Left(SubDomain):
    def inside(self, x, on_boundary): return on_boundary and near(x[0], 0.0, eps)

class Right(SubDomain):
    def inside(self, x, on_boundary): return on_boundary and near(x[0], 1.0, eps)

class Bottom(SubDomain):
    def inside(self, x, on_boundary): return on_boundary and near(x[1], 0.0, eps)

class Top(SubDomain):
    def inside(self, x, on_boundary): return on_boundary and near(x[1], 1.0, eps)

Left().mark(bmarks, 1); Right().mark(bmarks, 2)
Bottom().mark(bmarks, 3); Top().mark(bmarks, 4)

ds = Measure("ds", domain=mesh, subdomain_data=bmarks)

# Variational formulation
u, v = TrialFunction(V), TestFunction(V)
f, g = Constant((0.0, -9.81)), Constant((0.0, 0.0))
a = inner(grad(u), grad(v))*dx + dot(f, v)*ds(1) + dot(g, v)*ds(2)

# Boundary conditions
bc_top = DirichletBC(V, Constant((1.0, 0.0)), bmarks, 4)
bcs = [bc_top]

# Solve
u_soln = Function(V); solve(a == 0, u_soln, bcs=bcs)
p_soln = Function(V); solve(div(u) == 0, p_soln)

# Output
import matplotlib.pyplot as plt

plt.figure(); c = plot(abs(u_soln)); plt.colorbar(c)
plt.savefig("q5_speed.png")
plot(u_soln, mode="displacement", title="Velocity (u)"); plt.savefig("q5_u.png")
plot(p_soln, title="Pressure (p)"); plt.savefig("q5_p.png")