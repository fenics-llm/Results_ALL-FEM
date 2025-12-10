# filename: flag_motion.py

import numpy as np
from fenics import *
from ufl_legacy import inner, dot

# Define constants and parameters
rho_f = 1000  # kg/m^3
nu_f = 1e-3   # m^2/s
L = 2.5       # m
H = 0.41      # m
U_bar = 1     # m/s
t_final = 10  # s

# Create mesh and define the computational domain
mesh = create_unit_square(L, H)
cell_volume = CellVolume(mesh)

# Define the fluid domain (Ω_f) and the solid flag domain (Ω_s)
Omega_f = CellCentredMesh(cell_volume, mesh)
Omega_s = CellCentredMesh(cell_volume, mesh)

# Create function spaces for the fluid velocity (v_f), pressure (p_f), and solid displacement (u_s)
Vf = FunctionSpace(Omega_f, 'P', 1)
Pf = FunctionSpace(Omega_f, 'P', 2)
Us = FunctionSpace(Omega_s, 'P', 1)

# Define the boundary conditions
def inlet_bc(v):
    return v.subvalue(0) * (1 - np.cos(np.pi * t / 2)) / 2

def outlet_bc(p):
    return p.subvalue(L, 0)

bc_inlet = DirichletBC(Vf, inlet_bc, 'left')
bc_outlet = DirichletBC(Pf, outlet_bc, 'right')

# Define the boundary conditions for the solid flag
bc_top_bottom = DirichletBC(Us, 0, 'top') + DirichletBC(Us, 0, 'bottom')

# Create the fluid and solid problems
problem_f = NavierStokesProblem(Vf, Pf, Omega_f, rho_f, nu_f)
problem_s = StVenantKirchhoffProblem(Us, Omega_s)

# Solve the fluid problem using the Arbitrary Lagrangian Eulerian form of the Navier–Stokes equations
t = 0
while t < t_final:
    # Update the mesh and function spaces
    problem_f.update_mesh(mesh)
    problem_s.update_mesh(mesh)

    # Assemble the global stiffness matrix and right-hand side vector for the fluid problem
    Kf, b = problem_f.assemble_global_stiffness_matrix_and_rhs()

    # Solve the fluid problem using the Newton-Raphson method
    solution_f = solve(Kf, b)

    # Update the mesh and function spaces
    problem_s.update_mesh(mesh)
    problem_f.update_mesh(mesh)

    # Assemble the global stiffness matrix and right-hand side vector for the solid problem
    Ks, b = problem_s.assemble_global_stiffness_matrix_and_rhs()

    # Solve the solid problem using the Newton-Raphson method
    solution_s = solve(Ks, b)

    # Update the time step
    t += 0.01

# Save the fluid velocity and pressure fields and the beam displacement in XDMF format
problem_f.save('fluid_velocity.xdmf')
problem_f.save('pressure.xdmf')
problem_s.save('displacement.xdmf')

# Report the displacement components of point A with time
A = Point(0.60, 0.20)
print("Displacement components of point A at t=0:")
print(A.displacement())