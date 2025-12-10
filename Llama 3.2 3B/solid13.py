from fenics import *

# Create mesh
mesh = RectangleMesh(Point(0, 0), Point(1, 0.2), 64, 32)

# Define boundary conditions
def left_boundary(x):
    return (x[0], 0)

def hole_boundary(x):
    return (x[0] - 0.5 + 0.04 * cos(M_PI*x[0]), x[1])

def right_boundary(x):
    return (x[0] + 0, x[1])

def top_boundary(x):
    return (x[0], x[1] + 0)

def bottom_boundary(x):
    return (x[0], x[1] - 0.2)

# Define the function space
V = FunctionSpace(mesh, 'CG', 2)

# Create boundary conditions
left_bc = DirichletBC(V, Constant(0), 'left')
hole_bc = DirichletBC(V, Constant(-0.01), 'hole')
right_bc = DirichletBC(V, Constant(0), 'right')
top_bc = DirichletBC(V, Constant(0), 'top')
bottom_bc = DirichletBC(V, Constant(0), 'bottom')

# Define the test and trial functions
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(V)

# Compute the strain energy
W_p = (1/2) * integral(p**2 * dx)

# Compute the deformation gradient
F = Gradient(u)

# Compute the Cauchy stress tensor
sigma = 2*Epsilon(F)

# Compute the hydrostatic pressure
p = sigma[0, 0] - p

# Assemble the equations
eqn_p = W_p + integral(p * dx) - integral(1 * dx)
eqn_u = eqn_p - integral(sigma[0, 0] * n * dx)

# Apply boundary conditions
eqn_p.apply(left_bc)
eqn_p.apply(hole_bc)
eqn_p.apply(right_bc)
eqn_p.apply(top_bc)
eqn_p.apply(bottom_bc)

eqn_u.apply(left_bc)

# Solve the equations
u_expr = Expression('u_x', u=u)
p_expr = Expression('p', p=p)
F_expr = Expression('F', F=Gradient(u_expr))
sigma_expr = Expression('sigma', sigma=F_expr)

u_sol = Function(V)
p_sol = Function(V)

solve(eqn_u, u_sol, solver='CG')
solve(eqn_p, p_sol, solver='CG')

# Compute the von Mises equivalent stress
sigma_vm_x = sigma[0].get_value(mesh, 'left')
sigma_vm_y = sigma[1].get_value(mesh, 'left')

# Save the displacement field in XDMF format
u_x = u_sol.get_value(mesh, 'left')
u_y = u_sol.get_value(mesh, 'left')

with open('q13_def.xdmf', 'w') as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<Mesh>\n')
    f.write('  <vertices>\n')
    for i in range(mesh.num_vertices):
        x = mesh.coordinates[i]
        f.write('    <vertex>\n')
        f.write('      <x>{}</x>\n'.format(x[0]))
        f.write('      <y>{}</y>\n'.format(x[1]))
        f.write('    </vertex>\n')
    f.write('  </vertices>\n')
    f.write('  <cells>\n')
    for i in range(mesh.num_cells):
        cell = mesh.cells[i]
        f.write('    <cell>\n')
        f.write('      <c0>{}</c0>\n'.format(cell[0]))
        f.write('      <c1>{}</c1>\n'.format(cell[1]))
        f.write('      <c2>{}</c2>\n'.format(cell[2]))
        f.write('    </cell>\n')
    f.write('  </cells>\n')
    f.write('</Mesh>\n')

# Save the von Mises equivalent stress as a color map
import matplotlib.pyplot as plt

sigma_vm_x = sigma_vm_x.get_value(mesh, 'left')
sigma_vm_y = sigma_vm_y.get_value(mesh, 'left')

plt.imshow(sigma_vm_x, cmap='RdYlGn', origin='lower')
plt.colorbar()
plt.savefig('q13_vm.png')

# Save the displacement field
u_x.save('q13_def.png')