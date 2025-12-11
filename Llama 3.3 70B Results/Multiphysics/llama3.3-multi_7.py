# filename: coupled_flow.py
from fenics import *

# Define parameters
L = 1.0         # length of channel
H = 0.20        # height of channel
U_bar = 1.0     # inlet velocity
rho = 1.0       # density
mu = 0.01       # dynamic viscosity
K = 1e-6        # permeability

# Create mesh
mesh = RectangleMesh(Point(0, 0), Point(L, H), 100, 20)

# Define subdomains
class PorousFilter(SubDomain):
    def inside(self, x, on_boundary):
        return (x[0] >= 0.4 and x[0] <= 0.6)

class FreeFluid(SubDomain):
    def inside(self, x, on_boundary):
        return not (x[0] >= 0.4 and x[0] <= 0.6)

# Create subdomain markers
subdomains = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
porous_filter = PorousFilter()
free_fluid = FreeFluid()
porous_filter.mark(subdomains, 1)
free_fluid.mark(subdomains, 2)

# Define function spaces
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

# Define boundary conditions
u_inlet = Expression('6*U_bar*x[1]*(H-x[1])/H/H', U_bar=U_bar, H=H, degree=2)
bcs_u = DirichletBC(V, as_vector((u_inlet, 0)), 'on_boundary && near(x[0], 0)')
bcs_walls = [DirichletBC(V, Constant((0, 0)), 'on_boundary && near(x[1], 0.0)'),
             DirichletBC(V, Constant((0, 0)), 'on_boundary && near(x[1], H)')]
bcs = [bcs_u] + bcs_walls

# Define variational problem
u, v = TrialFunction(V), TestFunction(V)
p, q = TrialFunction(Q), TestFunction(Q)

a_ns = rho*inner(dot(u, nabla_grad(u)), v)*dx + mu*inner(grad(u), grad(v))*dx - inner(p, div(v))*dx
L_ns = inner(Constant((0, 0)), v)*dx

a_brinkman = -inner(p, div(v))*dx + mu*inner(grad(u), grad(v))*dx - (mu/K)*inner(u, v)*dx
L_brinkman = inner(Constant((0, 0)), v)*dx

# Solve variational problem for velocity and pressure separately
u_sol = Function(V)
p_sol = Function(Q)

F_u = rho*inner(dot(u, nabla_grad(u)), v)*dx + mu*inner(grad(u), grad(v))*dx 
F_p = inner(div(u), q)*dx

solve(F_u == 0, u_sol, bcs=bcs)
solve(F_p == 0, p_sol)

# Save velocity and pressure fields to file
with XDMFFile('q14_solution.xdmf') as f:
    f.write(u_sol, 0)
    f.write(p_sol, 0)

# Compute pressure drop across porous block
x_before = 0.4 - 1e-6
x_after = 0.6 + 1e-6
p_before = p_sol(x_before, H/2)
p_after = p_sol(x_after, H/2)
dp = p_before - p_after

# Save pressure drop to file
with open('q14_dp.txt', 'w') as f:
    f.write(str(dp))

# Save color map of velocity magnitude to file
import matplotlib.pyplot as plt
import numpy as np

u_magnitude = np.sqrt(u_sol.vector().get_local()**2)
plt.imshow(u_magnitude.reshape((20, 100)), cmap='viridis', extent=(0, L, 0, H))
plt.colorbar(label='Velocity Magnitude')
plt.savefig('q14_speed.png')
