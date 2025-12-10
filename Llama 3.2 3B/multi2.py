# filename: allen_cahn.py

import numpy as np
from fenics import *

# Define the domain
mesh = RectangleMesh(Point(0, 0), Point(1, 1), 100, 100)

# Create a finite element function space
V = FunctionSpace(mesh, 'P', 2)

# Define the boundary conditions
bc = DirichletBC(V, Constant((0.0, 0.0)), 'on_boundary')

# Initialize the phase field
def initialize_phi(x):
    return np.tanh(np.abs(d_rect(x[0], x[1])) / (np.sqrt(2) * epsilon))

phi = Function(V)
phi.interpolate(initialize_phi)

# Define the time stepping parameters
t_final = 0.20

# Create a time-stepping loop
for t in [0, 0.05, 0.10, t_final]:
    # Compute the time step size
    dt = min(dt, t_final - t) / len([t])

    # Assemble the problem
    u = Function(V)
    u.interpolate(1.0)

    # Define the variational form of the equation
    def F(v):
        return M * (np.mean((v**2 - 1)**2 / epsilon) - np.mean(np.dot(v, np.dot(grad(u), grad(u)))))

    # Compute the residual
    u_new = Function(V)
    u_new.interpolate(0.5*(u + u))
    F_u = F(u_new)

    # Solve for u using the Newton method
    def solve_u():
        u_new = Function(V)
        u_new.interpolate(0.5*(u + u))
        F_u = F(u_new)
        for i in range(10):
            v = Function(V)
            v.interpolate(1.0)
            J = inner(grad(v), grad(u_new))
            dJdu = inner(grad(v), grad(u_new))
            F_v = M * (np.mean((v**2 - 1)**2 / epsilon) - np.mean(np.dot(v, np.dot(grad(u_new), grad(u_new)))))
            F_u -= dt * J * F_v
            u_new += dt * J * v
        return u_new

    # Compute the solution
    u = solve_u()

    # Compute the phase field
    phi_new = Function(V)
    phi_new.interpolate(initialize_phi)
    for i in range(10):
        v = Function(V)
        v.interpolate(1.0)
        J = inner(grad(v), grad(phi_new))
        dJdphi = inner(grad(v), grad(phi_new))
        F_v = M * (np.mean((v**2 - 1)**2 / epsilon) - np.mean(np.dot(v, np.dot(grad(phi_new), grad(phi_new)))))
        phi_new += dt * J * v
    phi = phi_new

    # Output the phase field at each time step
    if t == 0.00:
        file_name = 't_{}_phi.xdmf'.format(t)
    elif t == 0.05:
        file_name = 't_{}_phi.xdmf'.format(t)
    elif t == 0.10:
        file_name = 't_{}_phi.xdmf'.format(t)
    else:
        file_name = 't_{}_phi.xdmf'.format(t)

    # Write the phase field to XDMF format
    with XDMFFile(file_name, mode='w') as f:
        f.write_mesh(mesh)
        for i in range(V.dim()):
            f.append_value(phi, i)

# Print the final solution
print("Final solution saved to t_0.20_phi.xdmf")