# filename: femics.py

from fenics import *

# Constants
E_Al = 70e9  # Pa
nu_Al = 0.30
E_Steel = 200e9  # Pa
nu_Steel = 0.30

# Geometry
Lx, Ly = 1.0, 0.20
dx, dy = Lx / 80, Ly / 16
cell_type = 'triangle'
mesh = UnitSquareMesh(80, 16, cell_type=cell_type)

# Material properties
material_properties = {
    'Al': {'E': E_Al, 'nu': nu_Al},
    'Steel': {'E': E_Steel, 'nu': nu_Steel}
}

# Define the function space for the displacement field
V = FunctionSpace(mesh, 'P', 1)

# Define the test function and trial function
u = TrialFunction(V)
v = TestFunction(V)

# Define the boundary conditions
bc_left = DirichletBC(V, Constant((0, 0)), 'left')
bc_right = DirichletBC(V, Constant((0, -5000)), 'right')
bc_top_bottom = NeumannBC('top_bottom', Constant(0), 'on_boundary')

# Assemble the stiffness matrix
K = assemble(linear_elasticity_modulus(material_properties) * inner(grad(u), grad(v)) * dx)

# Apply boundary conditions and loads
bc_left.apply(V)
bc_right.apply(V)
bc_top_bottom.apply(V)

# Solve for the displacement field using GMRES
u_solution = Function(V)
solve(K, u_solution, v, 'GMRES', 'nonlinear')

# Post-process the solution to save the displacement magnitude as a color map
q7_disp = Expression('sqrt(u_x**2 + u_y**2)', u=u_solution)
plot(q7_disp, filename='q7_disp.png')