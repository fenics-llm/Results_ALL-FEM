# filename: transport_equation.py
from fenics import *
import numpy as np
from mshr import Circle, generate_mesh

# Parameters
D = 1.0e-5  # diffusivity
kappa = 1.0e-4  # decay rate
s = 1.0e-4  # velocity
R0 = 0.05  # initial radius
T = 10.0  # total time
dt = 0.01  # time step

# Create mesh
domain = Circle(Point(0, 0), R0)
mesh = generate_mesh(domain, 100)

# Define function space
V = FunctionSpace(mesh, 'P', 1)

# Define test and trial functions
u = TrialFunction(V)
v = TestFunction(V)

# Define initial condition
u_n = interpolate(Constant(1.0), V)

# Define variational form
F = u*v*dx + dt*D*dot(grad(u), grad(v))*dx - (u_n + dt*kappa*u_n)*v*dx + dt*s*dot(grad(u), v)*dx

# Create XDMF file for output
xdmffile_u = XDMFFile("concentration.xdmf")
xdmffile_u.parameters["flush_output"] = True
xdmffile_u.parameters["rewrite_function_mesh"] = False

# Time-stepping
t = 0
while t < T:
    # Compute solution
    u = Function(V)
    solve(F == 0, u)

    # Save solution to file
    xdmffile_u.write(u, t)

    # Update previous solution
    u_n.assign(u)

    # Move mesh
    ALE.move(mesh, s)

    # Update time
    t += dt

# Hold plot
interactive()