# filename: cahn_hilliard.py

import numpy as np
from fenics import *
from ufl import inner, grad, dot

# Parameters
theta = 1.5
alpha = 3000
T = 0.04
cbar = 0.63
r_min = -0.05
r_max = 0.05
dx = 1e-2
dy = 1e-2

# Create mesh and finite element function space
mesh = UnitSquareMesh(dx, dy)
fe_space = FunctionSpace(mesh, 'P', 1)

# Define boundary conditions
def periodic_bc(x):
    return x[0] % 1 == 0 or x[1] % 1 == 0

bc_c = DirichletBC(fe_space, Constant(0), 'on_boundary')
bc_mu = DirichletBC(fe_space, Constant(0), 'on_boundary')

# Define initial condition
def r(x):
    return np.random.uniform(r_min, r_max)

c_0 = Expression('cbar + r(x[0], x[1])', r=r)
c_0_v = c_0.vector()

# Define mobility function
def M(c):
    return c * (1 - c)

# Define time-stepping parameters
dt = 1e-7
t_max = T

# Create time-stepping loop
t = 0
while t < t_max:
    # Compute time step size
    dt_new = min(dt, t_max - t) / 10.0
    if abs(dt_new - dt) > 1e-6 * dt_new:
        print("Warning: Time step size reduced.")
    dt = dt_new

    # Solve Cahn-Hilliard equation
    mu = Expression('3*alpha*(cbar/(theta)*np.log(c/(1-cbar)) + 1 - 2*c)', c=c_0)
    v = Function(fe_space)
    u = Function(fe_space)
    w = Function(fe_space)

    def L(u, v):
        return inner(M(u) * grad(v), grad(u))

    def R(u, v):
        return inner(mu * M(u) * grad(v), grad(u)) + inner(grad(u), grad(u))

    F = R(c_0, u) - L(c_0, u)
    solve(F == 0, u)

    # Compute concentration and chemical potential
    c_t = Function(fe_space)
    mu_t = Function(fe_space)

    def compute_fields():
        nonlocal c_t, mu_t
        c_t.vector()[:] = c_0.vector()
        for i in range(1):
            mu_t.vector()[:] = mu(c_t).vector()

    compute_fields()

    # Compute time derivative of concentration and chemical potential
    def dt_dC_dt(u, v):
        return inner(M(u) * grad(v), grad(u))

    def dt_dMu_dt(u, v):
        return inner(mu * M(u) * grad(v), grad(u)) + inner(grad(u), grad(u))

    F_C = dt_dC_dt(c_0, c_t)
    F_Mu = dt_dMu_dt(c_0, mu_t)

    # Solve time derivative equations
    solve(F_C == 0, c_t)
    solve(F_Mu == 0, mu_t)

    # Update concentration and chemical potential
    c_0.vector()[:] += dt * c_t.vector()
    mu_0.vector()[:] = mu(c_0).vector()

    # Compute time step size for next iteration
    t += dt

# Save fields to XDMF file
file_c = File('cahn_hilliard.xdmf')
file_mu = File('mu.cahn_hilliard.xdmf')

for i in [0, 3e-6, 1e-4, 1e-3, 4e-2]:
    c_0.vector()[:] = cbar + r(np.random.uniform(0, 1), np.random.uniform(0, 1))
    mu_0.vector()[:] = 3 * alpha * (cbar / theta) * np.log(cbar / (1 - cbar)) + 1 - 2 * cbar
    file_c << c_0
    file_mu << mu_0

# Save fields to time-series file
file_ts = File('cahn_hilliard.xdmf')
for i in [0, 3e-6, 1e-4, 1e-3, 4e-2]:
    c_0.vector()[:] = cbar + r(np.random.uniform(0, 1), np.random.uniform(0, 1))
    mu_0.vector()[:] = 3 * alpha * (cbar / theta) * np.log(cbar / (1 - cbar)) + 1 - 2 * cbar
    file_ts << c_0
    file_mu << mu_0

print("Simulation completed.")
