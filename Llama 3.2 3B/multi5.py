#filename: fluid_structure_interaction.py

import numpy as np
from fenics import *

# Define constants
mu_f = 0.003  # Poise
rho_f = 1     # g·cm⁻³
rho_s = 1.1   # g·cm⁻³
nu_s = 0.49   # Poisson ratio
E_s = 3e5     # Young's modulus Pa
Delta_t = 1e-4  # s

# Define mesh and boundary conditions
mesh = UnitSquareMesh(100, 100)
x = MeshFunction("double", mesh)

def boundary(x):
    return x[0] == 0 or x[0] == 6 or x[1] == -0.1 or x[1] == 1.1

x.set_all(0)

# Define function spaces
Vf = FunctionSpace(mesh, 'P', 2)
Vs = FunctionSpace(mesh, 'P', 2)

# Define boundary conditions
bc_inlet = DirichletBC(Vf, Constant((-(2e4)/2 * (1 - np.cos(np.pi*t/2.5e-3)), 0)), x.dual(x))
bc_outlet = DirichletBC(Vf, Constant((0, 0)), x.dual(x))

# Define initial conditions
u0_f = Function(Vf)
v0_f = TestFunction(Vf)

u0_s = Function(Vs)
v0_s = TestFunction(Vs)

u0_f[0] = u0_f[1] = 0
u0_s[0] = u0_s[1] = 0

# Define Navier-Stokes equations for fluid
def navier_stokes(u, v):
    return (v * rho_f * u) + (mu_f * Laplace(u)) - (rho_f * g)

# Define solid mechanics equations
def solid_mechanics(u, v):
    return (v * E_s * nu_s * u) + (E_s * (1 - 2*nu_s) * u)

# Define time-stepping loop
t = 0
while t < 0.01:
    # Solve Navier-Stokes equations for fluid
    u_f = Function(Vf)
    v_f = TestFunction(Vf)
    F = navier_stokes(u_f, v_f)
    solve(F == 0, u_f)

    # Solve solid mechanics equations for solid
    u_s = Function(Vs)
    v_s = TestFunction(Vs)
    F = solid_mechanics(u_s, v_s)
    solve(F == 0, u_s)

    # Update velocity and displacement
    u_f[0] += u_f[1]
    u_s[0] += u_s[1]

    # Apply boundary conditions
    bc_inlet.apply(u_f)
    bc_outlet.apply(u_f)
    bc_inlet.apply(u_s)
    bc_outlet.apply(u_s)

    # Update time step
    t += Delta_t

# Save output velocity and displacement at time 0.005s and 0.1s in XDMF format
u_f_05 = u_f.copy()
u_f_1 = u_f.copy()

u_s_05 = u_s.copy()
u_s_1 = u_s.copy()

xdmf_file = File("output.xdmf")
xdmf_file << FunctionFile(u_f_05, name="fluid_velocity_0.005s") << FunctionFile(u_f_1, name="fluid_velocity_0.1s")
xdmf_file << FunctionFile(u_s_05, name="solid_displacement_0.005s") << FunctionFile(u_s_1, name="solid_displacement_0.1s")

# Print final velocity and displacement
print("Final fluid velocity:", u_f[0].eval()
      ,u_f[1].eval())
print("Final solid displacement:", u_s[0].eval(),u_s[1].eval())

