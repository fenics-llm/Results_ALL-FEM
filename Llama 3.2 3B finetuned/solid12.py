 filename: q12.py

from dolfin import *
import mshr
import matplotlib.pyplot as plt

mesh = mshr.RectangleMesh(Point(0, 0), Point(1, 0.2), 64, 32)
hole = mshr.Circle(Point(0.5, 0.1), 0.04)
mesh = mesh - hole
V = VectorFunctionSpace(mesh, "P", 1)

def eps(u): return sym(grad(u))
def sigma(u): return E*eps(u)/(2*(1+nu))+p*I

# Material parameters
E, nu = 5e3, 0.5
f = Constant((0.0, 0.0))

# Variational formulation
u, v = TrialFunction(V), TestFunction(V)
a = (E*eps(u)[0,1]*eps(v)[0,1] + E*eps(u)[1,0]*eps(v)[1,0])/(2*(1+nu)) - p*v*dx

# Boundary conditions
bc_left = DirichletBC(V, Constant((0.0, 0.0)), "near(x[0], 0.0)")
bc_right = DirichletBC(V, Constant((0.06, 0.0)), "near(x[0], 1.0)")

# Variational problem
p = Function(V)
solve(a == f, p, [bc_left, bc_right])

# Compute and save von Mises stress
sigma_u = sigma(u)
vm = sqrt(0.5*(sigma_u[0,0]**2 + sigma_u[0,1]**2 +
              sigma_u[1,0]**2 + sigma_u[1,1]**2 -
              2*sigma_u[0,0]*sigma_u[0,1] - 2*sigma_u[1,0]*sigma_u[1,1]))
vm_file = File("q12_vm.pvd")
vm_file << vm

# Compute and save hydrostatic pressure
p_file = File("q12_p.pvd")
p_file << p

plt.figure()
c = plot(p)
plt.colorbar(c)
plt.savefig("q12_p.png")

plt.figure()
c = plot(vm)
plt.colorbar(c)
plt.savefig("q12_vm.png")
