# filename: q12.py

from fenics import *

# Define the parameters of the problem
L = 2.0  # Length of the channel (m)
H = 0.20  # Height of the channel (m)
U_bar = 1.0  # Inlet velocity (m/s)
n = 0.5  # Flow behavior index
mu_0 = 0.5  # Consistency index (Pa·s^n)
rho = 1.0  # Density (kg/m^3)

# Define the mesh
mesh = UnitSquareMesh(240, 24)

# Define the finite element function space
V = FunctionSpace(mesh, 'P', 2)

# Define the test and trial functions for velocity and pressure
u, p = TrialFunction(V), TrialFunction(V)
v_u, v_p = TestFunction(V), TestFunction(V)

# Compute the strain-rate tensor
D = (grad(u) + grad(u).T)/2

# Compute the effective viscosity
mu_eff = mu_0 * (abs(D)**(n-1))

# Define the boundary conditions
bc_inlet = DirichletBC(V, u*U_bar*(H-y)/(H**2), 0)
bc_walls = DirichletBC(V, 0, 0.0)
bc_outlet = NeumannBC(V, -p*I, 0)

# Define the variational problem
problem = Problem(V, bc_inlet, bc_walls, bc_outlet, p*v_p, u*v_u, rho*(u*grad(v_u) - v_u*grad(u)))

# Compute the solution
solution = problem.solve()

### Step 3: Post-processing and output

# Extract the velocity magnitude
velocity_magnitude = abs(solution[0])

# Save the color map of the velocity magnitude
plot(velocity_magnitude, cmap='viridis', filename='q12_speed.png')

# Extract the streamwise velocity profile at mid-channel
ux_profile = solution[0].get_value((1.0, 0.5), 0)

# Save the velocity profile to a file
np.savetxt('q12_profile.csv', ux_profile)

# Compute the maximum velocity
max_velocity = np.max(velocity_magnitude)

### Step 4: Report the results
print(f'Maximum velocity at x = L/2: {max_velocity} m/s')