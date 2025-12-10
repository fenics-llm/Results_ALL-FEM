from fenics import UnitSquareMesh, Function, DirichletBC, File, NavierStokesProblem

# Create a mesh for the upstream channel
mesh_upstream = UnitSquareMesh(-3*1.0, -3*1.0, 20*1.0, 2*1.0)

# Define the inlet boundary condition for velocity
u_inlet = Function('VectorField', (mesh_upstream, mesh_upstream.H))
u_inlet[0] = 6 * 1.0 * (mesh_upstream.y/mesh_upstream.H) * (1 - mesh_upstream.y/mesh_upstream.H)
u_inlet[1] = 0

# Define the solid wall boundary conditions for velocity
bottom_wall_bc = DirichletBC(mesh_upstream, u_inlet, 'on_boundary')
top_wall_bc = DirichletBC(mesh_upstream, [0.0, 0.0], 'on_boundary')

# Define the outlet boundary condition for traction
outlet_bc = File(mesh_upstream, "velocity.pvd")

# Define the velocity and pressure variables
u = Function('VectorField', (mesh_upstream, mesh_upstream.H))
p = Function('ScalarField', (mesh_upstream))

# Assemble the weak form of the Navier-Stokes equations
problem = NavierStokesProblem(mesh_upstream, u, p)

# Solve the problem
problem.solve()

# Define the velocity gradient variable
ux = u[0]

# Compute the wall shear stress on the top wall
tau_w = Function('ScalarField', (mesh_upstream, mesh_upstream.H))
tau_w[0] = 1.0 * 0.01 * (ux / mesh_upstream.H).compute_at((20*1.0/2.0, 0))

# Save the velocity field as a PNG file
vtk_file = File("q6_u.pvd")
vtk_file << u

# Save the solution fields in XDMF format
xdmf_file = File("q6_soln.xdmf")
xdmf_file << (u, p)

# Terminate
print('Done')
