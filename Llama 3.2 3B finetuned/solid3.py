# filename: q3.py

from dolfin import *
import mshr
import matplotlib.pyplot as plt

mesh = mshr.RectangleMesh(mshr.Point(0, 0), mshr.Point(1.0, 0.20),
                          64, 32)
hole_radius = 0.05
circle = mshr.Circle(Point(0.50, 0.10), hole_radius)
domain = mesh - circle
V = FunctionSpace(domain, "CG", 2)

E = 200000.0
nu = 0.3

def sigma(u):
    return E * ((1 + nu) * sym(grad(u)) - nu * tr(sym(grad(u))) * I)

bc_left = DirichletBC(V, Constant((0.0, 0.0)), "on_boundary && x[0] == 0.0")
u = TrialFunction(V)
v = TestFunction(V)
a = sigma(u) * grad(v) * dx
L = Constant((2.0e6, 0.0)) * v * dx

bc_right = DirichletBC(V, Constant((0.0, 0.0)), "on_boundary && x[0] == 1.0")

u_sol = Function(V)
solve(a == L, u_sol, [bc_left, bc_right])

vmises = sqrt(0.5 * (pow(sigma(u_sol)[0, 0] - sigma(u_sol)[1, 1],
                     2) + 3 * pow(sigma(u_sol)[0, 1], 2)))

plt.figure()
p = plot(vmises)
plt.colorbar(p)
plt.title("von Mises stress")
plt.savefig("q3_vm.png")

# save displacement
File("q3_u.xdmf") << u_sol

print("Maximum von Mises stress at hole boundary:", vmises.max())
print("Stress Concentration Factor (Kt):", 2.0e6 / vmises.max())