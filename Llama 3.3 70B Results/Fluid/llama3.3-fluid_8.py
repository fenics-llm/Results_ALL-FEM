from fenics import *

# Create mesh
mesh = RectangleMesh(Point(0, 0), Point(1, 0.20), 128, 32)

# Define function spaces
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

# Define trial and test functions
u, v = TrialFunction(V), TestFunction(V)
p, q = TrialFunction(Q), TestFunction(Q)

# Define parameters
rho = 1.0
mu = 0.01
G = 1.0

# Define body force
f = Constant((G, 0))

# Define boundary conditions
def boundary_y0(x, on_boundary):
    return on_boundary and near(x[1], 0)

def boundary_yH(x, on_boundary):
    return on_boundary and near(x[1], 0.20)

u_bc_y0 = DirichletBC(V, Constant((0, 0)), boundary_y0)
u_bc_yH = DirichletBC(V, Constant((0, 0)), boundary_yH)

# Define variational problem
F_u = rho * inner(grad(u), grad(v)) * dx + mu * inner(div(u), div(v)) * dx - inner(f, v) * dx 
F_p = inner(p, q) * dx + inner(q, div(u)) * dx

# Solve the Navier-Stokes equations
u_sol = Function(V)
p_sol = Function(Q)

a_u = lhs(F_u)
L_u = rhs(F_u)
solve(a_u == L_u, u_sol, [u_bc_y0, u_bc_yH])

a_p = lhs(F_p)
L_p = rhs(F_p)
solve(a_p == L_p, p_sol)

# Save solution to file
u_file = XDMFFile('q9_soln.xdmf')
u_file.write(u_sol, 0)
p_file = XDMFFile('q9_soln.xdmf')
p_file.write(p_sol, 0)