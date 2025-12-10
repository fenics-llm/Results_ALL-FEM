# filename: vms_cylinder.py

import numpy as np
from dolfin import *

# Define parameters
U = 1.0  # m/s
nu = 2.56e-5  # m²/s
rho = 1.0  # kg/m³
D = 1.0  # m
t_start = 8.0  # s
t_end = 10.0  # s

# Create mesh and define function spaces
mesh = UnitSquareMesh(120, 120)
V = FunctionSpace(mesh, 'P', 2)

# Define boundary conditions
def inflow(x):
    return (U, 0)

bc_inflow = DirichletBC(V, Expression(inflow), 'left')
bc_outflow = DirichletBC(V, (0, 0), 'right')

bc_top_bottom = DirichletBC(V, (0, 0), 'top')
bc_bottom = DirichletBC(V, (0, 0), 'bottom')
bc_cylinder = DirichletBC(V, (0, 0), 'cylinder')

# Define variational multiscale formulation
def residual(u, v):
    return inner(grad(u) - grad(v), grad(v)) + nu * inner(div(u), div(v))

# Define stabilizations
def sw_petrov_galerkin(u, v):
    return inner(grad(u), grad(v)) + 0.5 * inner(grad(u) * (grad(v) * n), dS)

def sw_grad_div(u, v):
    return inner(grad(u), grad(v))

# Create variational multiscale problem
problem = VariationalMultiscaleProblem(
    residual,
    bc_inflow,
    bc_outflow,
    bc_top_bottom,
    bc_bottom,
    bc_cylinder,
    sw_petrov_galerkin,
    sw_grad_div
)

# Define initial condition and perturbation (if desired)
u0 = Function(V)
p0 = Function(V)

# Initialize u0 and p0 with initial conditions
u0[:] = 0.0
p0[:] = 0.0

# Optional small perturbation to trigger vortex shedding
# u0 += 1e-3 * sin(pi*x[0]) * cos(pi*y[1])

# Solve the problem
t = 0.0
dt = 0.01
while t < t_end:
    # Update solution
    u, p = solve(problem, u0, p0)
    
    # Compute mean drag coefficient
    D_mean = compute_mean_drag_coefficient(u, p, U, rho, nu, D, t_start, t_end)
    
    # Print mean drag coefficient
    print(f"Mean drag coefficient at t={t:.2f}s: {D_mean:.4f}")
    
    # Save final velocity and pressure fields
    vms_solution = save_final_velocity_and_pressure_fields(u, p, u0, p0, V, mesh)
    
    # Update time
    t += dt

# Save final solution to file
save_final_solution(vms_solution, 'vms_solution.xdmf')

def compute_mean_drag_coefficient(u, p, U, rho, nu, D, t_start, t_end):
    # Compute drag force
    F_d = 0.5 * rho * U**2 * D
    
    # Compute mean drag coefficient
    D_mean = F_d / (rho * U**2)
    
    return D_mean

def save_final_velocity_and_pressure_fields(u, p, u0, p0, V, mesh):
    # Save final velocity field
    vms_solution_velocity = File('vms_solution_velocity.pvd')
    vms_solution_velocity << u
    
    # Save final pressure field
    vms_solution_pressure = File('vms_solution_pressure.pvd')
    vms_solution_pressure << p
    
    return vms_solution_velocity, vms_solution_pressure

def save_final_solution(vms_solution_velocity, vms_solution_pressure, filename):
    vms_solution_velocity.write_postscript(filename)
    vms_solution_pressure.write_postscript(filename)
