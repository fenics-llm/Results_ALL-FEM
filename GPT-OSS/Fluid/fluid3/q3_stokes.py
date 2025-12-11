# filename: q3_stokes.py
from dolfin import *
import matplotlib.pyplot as plt

# Parameters
mu = Constant(1.0)   # dynamic viscosity
rho = Constant(1.0)  # density (unused for Stokes)

# Mesh
nx, ny = 96, 96
mesh = UnitSquareMesh(nx, ny)

# Taylor–Hood elements (P2 velocity, P1 pressure)
V_el = VectorElement('Lagrange', mesh.ufl_cell(), degree=2)
P_el = FiniteElement('Lagrange', mesh.ufl_cell(), degree=1)
W_el = MixedElement([V_el, P_el])
W = FunctionSpace(mesh, W_el)

# Boundary definitions
tol = 1E-14
class Lid(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 1.0, tol)

class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[0], 0.0, tol) or
                                near(x[0], 1.0, tol) or
                                near(x[1], 0.0, tol))

lid = Lid()
walls = Walls()

# Dirichlet BCs for velocity
bc_lid   = DirichletBC(W.sub(0), Constant((1.0, 0.0)), lid)
bc_walls = DirichletBC(W.sub(0), Constant((0.0, 0.0)), walls)
bcs = [bc_lid, bc_walls]

# Trial and test functions
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

# Variational form (steady Stokes)
a = (mu*inner(grad(u), grad(v)) - div(v)*p + q*div(u))*dx
L = Constant(0.0)*dot(v, Constant((0.0, 0.0)))*dx  # RHS = 0

# Solve linear system
w = Function(W)
solve(a == L, w, bcs, solver_parameters={'linear_solver': 'mumps'})

# Split solution
(u_sol, p_sol) = w.split()

# Compute speed magnitude |u|
V = FunctionSpace(mesh, 'Lagrange', 2)
speed = project(sqrt(dot(u_sol, u_sol)), V)

# Save speed plot
plt.figure(figsize=(6,5))
p = plot(speed, title='Speed magnitude |u|')
plt.colorbar(p)
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.savefig('q3_speed.png', dpi=300)

# Save solution to XDMF
with XDMFFile('q3_soln.xdmf') as xdmf:
    xdmf.write(u_sol, 0.0)
    xdmf.write(p_sol, 0.0)