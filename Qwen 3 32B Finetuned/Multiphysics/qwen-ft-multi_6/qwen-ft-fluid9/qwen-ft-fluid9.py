from dolfin import *
from mshr import Rectangle, Circle, generate_mesh
import numpy as np

# Geometry
channel = Rectangle(Point(0, 0), Point(2.5, 0.41))
flag = Rectangle(Point(0.6, 0.19), Point(0.95, 0.21))
pole = Circle(Point(0.2, 0.2), 0.05)
mesh = generate_mesh(channel - flag - pole, 64)

# Function spaces
V = VectorFunctionSpace(mesh, "P", 2)
Q = FunctionSpace(mesh, "P", 1)
ME = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, ME)
V_s = VectorFunctionSpace(mesh, "P", 2)

# Parameters
rho_f, nu_f = 1000.0, 1e-3
rho_s, mu_s, nu_s = 1000.0, 5e5, 0.4
lambda_s = mu_s * nu_s / (1 - 2*nu_s)

# Variational forms
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
(u_s, p_s) = TrialFunctions(W)
(v_s, q_s) = TestFunctions(W)
f = Constant((0.0, 0.0))
a_fluid = rho_f*dot(u, v)*dx + rho_f*nu_f*inner(grad(u), grad(v))*dx - p*div(v)*dx - q*div(u)*dx
L_fluid = rho_f*dot(f, v)*dx

# Boundary conditions
inlet = Expression(("1.5*0.2*x[1]*(0.41 - x[1])/(0.41/2.0*0.41/2.0)", "0.0"), degree=2)
bcs = [DirichletBC(W.sub(0), inlet, "near(x[0], 0.0)"),
       DirichletBC(W.sub(0), Constant((0.0, 0.0)), "near(x[0], 2.5)"),
       DirichletBC(W.sub(0), Constant((0.0, 0.0)), "near(x[1], 0.0) || near(x[1], 0.41)"),
       DirichletBC(W.sub(0), Constant((0.0, 0.0)), "on_boundary && (x[0] > 0.2 - 0.05 && x[0] < 0.2 + 0.05 && x[1] > 0.2 - 0.05 && x[1] < 0.2 + 0.05)")]

# Solve
w = Function(W)
solve(a_fluid == L_fluid, w, bcs)
u_sol, p_sol = w.split()

# Save
file = XDMFFile("fsi_results.xdmf")
file.write(u_sol, 0.0)
file.write(p_sol, 0.0)

# Point A displacement
A = Point(0.6, 0.2)
print("Displacement at A:", u_sol(A))