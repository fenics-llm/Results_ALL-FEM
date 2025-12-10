# filename: q13.py

from dolfin import *
import mshr
import matplotlib.pyplot as plt

def plot_def(mesh):
    def_plot = mesh
    def_plot = Circle(Point(0.5, 0.1), 0.04)
    def_plot = def_plot - Rectangle(Point(0, 0), Point(1, 0.2))
    return Plot(def_plot, title="q13: deformed geometry")

def plot_vm(mesh):
    vm = Expression("sqrt(0.5*((pow(x[0]-0.5,2)+pow(x[1]-0.1,2))-0.04*0.04))*1000/10", degree=3)
    return Plot(vm, title="q13: von Mises stress")

mesh = mshr.generate_mesh(mshr.Rectangle(Point(0, 0), Point(1, 0.2)), 64)
u = TrialFunction(V)
v = TestFunction(V)
E, nu = 5e6, 0.5
f = Constant((0.0, -9.81))
def W(u):
    return E*inner(u,u)/2 + p*u[1]*dx

a = inner(grad(u), grad(v))*dx + f*v*dx
L = dot(f,v)*dx

p = Function(V)
solve(a == L, p)

def_plot()
plot_vm()
plt.show()

