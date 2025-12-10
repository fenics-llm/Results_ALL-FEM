from dolfin import *
import mshr

# 1. Geometry: rectangular plate with semicircular notch
Omega = Rectangle(Point(0, 0), Point(1, 0.2)) - Circle(Point(0.5, 0.2), 0.05)
mesh = mshr.generate_mesh(Omega, 64)

# 2. Material: E=200 GPa, nu=0.3
E, nu = 200e9, 0.3

# 3. Boundary conditions
bdr = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
bdr.set_all(0)

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0)

Bottom().mark(bdr, 1)   # bottom: fixed

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], 0.2, DOLFIN_EPS)
                                and (x[0] >= 0.45 or x[0] <= 0.55))

Top().mark(bdr, 3)      # top: -10 MPa y

class Vertical(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0)

Vertical().mark(bdr, 2)  # vertical sides: traction-free

class NotchArc(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[0], 0.5, DOLFIN_EPS)
                                and near(sqrt((x[1]-0.2)**2 + x[0]**2), 0.05))

NotchArc().mark(bdr, 4) # notch arc: traction-free

# 4. Variational formulation
V = VectorFunctionSpace(mesh, "P", 1)
u = TrialFunction(V); v = TestFunction(V)

f = Constant((0.0, -10.0))   # top boundary condition
a = inner(sigma(u), sigma(v))*dx
L = dot(f, v)*dx

# 5. Solve displacement field u
w = Function(V)
solve(a == L, w)

# 6. Compute von Mises stress and save to file
sigma = project(sym(eigensystem(w))[0], VectorFunctionSpace(mesh,"P",1))
vmises = project(sqrt(0.5*((sigma[0,0]-sigma[1,1])**2 + (sigma[0,1]+sigma[1,0])**2)), FunctionSpace(mesh,"P",1))

# 7. Save results
file_vm = File("q6_vm.pvd")
vmises.compute_vertex_values()
file_vm << vmises