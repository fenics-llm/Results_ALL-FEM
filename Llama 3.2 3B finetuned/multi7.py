# filename: q14.py

from dolfin import *
import mshr
import matplotlib.pyplot as plt

# Geometry
domain = mshr.Rectangle(Point(0, 0), Point(1.0, 0.20))
filter = mshr.Rectangle(Point(0.4, 0), Point(0.6, 0.20))
mesh = mshr.generate_mesh(domain - filter, 64)
V = VectorFunctionSpace(mesh, "P", 1)

# Model
mu = Constant(0.01) # Pa s
K   = Constant(1e-6) # m²

def eps(u): return sym(grad(u))
def div(u): return dot(grad(u), dx)

u_f = Expression(("6*U_bar*y*(H-y)/H^2", "0.0"), U_bar=1.0, H=0.20, degree=2)
bcs = [DirichletBC(V, u_f, "near(x[1], 0.0)"),
       DirichletBC(V, Constant((0.0, 0.0)), "on_boundary && near(x[1], 0.2)")]

u, p = TrialFunctions(V), TestFunctions(V)
a = (mu*eps(u)-p*dx)*eps(w)+K*u*w*dx + div(u)*w*dx
L = Constant(0.0)*w*dx

# Solve
u_sol = Function(V); solve(a == L, u_sol, bcs)

# Output
plt.figure(); c = plot(abs(u_sol)); plt.colorbar(c)
plt.savefig("q14_speed.png")

p_vec = project(p, VectorFunctionSpace(mesh,"P",1))
print("Pressure drop: ", p_vec(Point(0.6, 0.2)) - p_vec(Point(0.4, 0.2)))

# Save
File("q14_solution.pvd") << u_sol
File("q14_solution.xml") << u_sol
File("q14_solution.h5") << u_sol