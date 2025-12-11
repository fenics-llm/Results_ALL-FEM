from fenics import *
import numpy as np

# Create mesh
L = 2.0
H = 0.20
mesh = RectangleMesh(Point(0, 0), Point(L, H), 100, 10)

# Define function spaces
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

# Define boundary conditions
u_inlet = Constant((0, 0))
p_in = Constant(1.0)
p_out = Constant(0.0)

def boundary(x, on_boundary):
    return on_boundary

bc_u = DirichletBC(V, u_inlet, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

mu = Constant(1.0)
rho = Constant(1.0)

a_u = mu * inner(grad(u), grad(v)) * dx 
L_inlet = -inner(p_in, dot(v, FacetNormal(mesh))) * ds
L_outlet = -inner(p_out, dot(v, FacetNormal(mesh))) * ds

# Define the pressure equation
a_p = inner(grad(p), grad(q)) * dx + inner(div(u), q) * dx 

# Compute solution
u_sol = Function(V)
p_sol = Function(Q)

A_u = assemble(a_u)
bcs = [bc_u]
[bc.apply(A_u) for bc in bcs]

# Apply boundary conditions to the right-hand side of the equation
b = assemble(L_inlet + L_outlet)
[bc.apply(b) for bc in bcs]

solve(A_u, u_sol.vector(), b)

# Solve pressure equation
A_p = assemble(a_p)
b_p = assemble(inner(Constant(0), q) * dx)
solve(A_p, p_sol.vector())

# Save solution to file
vtkfile_u = File('u_sol.pvd')
vtkfile_p = File('p_sol.pvd')
vtkfile_u << u_sol
vtkfile_p << p_sol

# Save color map of the speed |u| over Ω to 'speed.png'
import matplotlib.pyplot as plt

# Get the speed values at each point in the mesh
speed_values = np.zeros(len(mesh.coordinates()))
for i, point in enumerate(mesh.coordinates()):
    speed_values[i] = np.linalg.norm(u_sol.at(point))

# Create a color map of the speed values
plt.imshow(speed_values.reshape((10, 100)), cmap='viridis', origin='lower')
plt.colorbar(label='Speed |u|')
plt.title('Color map of the speed |u| over Ω')
plt.savefig('speed.png')