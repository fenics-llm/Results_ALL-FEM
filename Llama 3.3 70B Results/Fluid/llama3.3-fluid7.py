# filename: navier_stokes.py
from fenics import *
import numpy as np

# Define parameters
R = 0.05  # radius of circular hole
H = 0.41  # height of channel
U_bar = 0.2  # mean inlet velocity
mu = 0.001  # dynamic viscosity
rho = 1.0  # density

# Create mesh
mesh = RectangleMesh(Point(0, 0), Point(2.2, H), 100, 20)

# Define function spaces
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

# Define boundary conditions
u_inlet = Expression(('6*U_bar*x[1]*(H-x[1])/(H*H)', '0'), U_bar=U_bar, H=H, degree=2)

def on_inlet(x, on_boundary):
    return on_boundary and near(x[0], 0)

def on_walls(x, on_boundary):
    return on_boundary and (near(x[1], 0) or near(x[1], H))

def on_circle(x, on_boundary):
    return on_boundary and ((x[0] - 0.2)**2 + (x[1] - 0.2)**2 < R**2)

bc_u_inlet = DirichletBC(V, u_inlet, on_inlet)
bc_u_walls = DirichletBC(V, Constant((0, 0)), on_walls)
bc_u_circle = DirichletBC(V, Constant((0, 0)), on_circle)

bcs_u = [bc_u_inlet, bc_u_walls, bc_u_circle]
bcs_p = []

# Define variational problem
u = TrialFunction(V)
p = TrialFunction(Q)
v = TestFunction(V)
q = TestFunction(Q)

a_u = mu * inner(grad(u), grad(v)) * dx + rho * inner(dot(u, nabla_grad(u)), v) * dx 
L_u = Constant(0) * v[0] * dx

a_p = inner(div(u), q) * dx
L_p = Constant(0) * q * dx

# Compute solution
u_sol = Function(V)
p_sol = Function(Q)

solve(a_u == L_u, u_sol, bcs_u)
solve(a_p == L_p, p_sol, bcs_p)

# Compute drag force on circle
n = FacetNormal(mesh)
F_D = assemble(-mu * inner(grad(u_sol), n) * u_sol * ds(6))

# Compute drag coefficient
D = 2 * R
C_D = 2 * F_D / (rho * U_bar**2 * D)

print(f"Drag coefficient C_D: {C_D}")

# Save color map of speed |u|
speed = sqrt(u_sol[0]**2 + u_sol[1]**2)
vtkfile_speed = File('q7_speed.pvd')
vtkfile_speed << speed

# Save velocity and pressure fields
vtkfile_soln_u = XDMFFile('q7_soln_u.xdmf')
vtkfile_soln_p = XDMFFile('q7_soln_p.xdmx')
vtkfile_soln_u.write(u_sol, 0)
vtkfile_soln_p.write(p_sol, 0)

# Save color map of speed as png image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(project(speed, Q).compute_vertex_values().reshape((mesh.num_cells(), 1)), cmap='viridis', extent=(0, 2.2, 0, H))
plt.colorbar(label='Speed')
plt.savefig('q7_speed.png')
