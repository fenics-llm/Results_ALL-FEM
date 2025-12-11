from fenics import *
import numpy as np

# Define parameters
U = 1.0  # m/s
nu = 2.56e-5  # m^2/s
rho = 1.0  # kg/m^3
D = 1.0  # m
T = 10.0  # s

# Create mesh
mesh = RectangleMesh(Point(-30*D, -30*D), Point(30*D, 30*D), 100, 100)

# Define function spaces
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

# Define boundary conditions
u_inflow = Constant((U, 0))
def boundary(x, on_boundary):
    return on_boundary and (x[0] < -29*D or abs(x[1]) > 29*D)

bc_u = DirichletBC(V, u_inflow, boundary)
class MyBC(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1e-5
        return on_boundary and near(x[0], 30*D, tol)

my_bc = MyBC()
bc_p = DirichletBC(Q, Constant(0), my_bc)

# Define variational formulation
u = TrialFunction(V)
p = TrialFunction(Q)
v = TestFunction(V)
q = TestFunction(Q)

F_u = rho*dot(u, v)*dx + nu*inner(grad(u), grad(v))*dx - p*div(v)*dx 
F_p = q*div(u)*dx

# Split the system into two separate equations
a_u = inner(u, v)*dx + nu*inner(grad(u), grad(v))*dx
L_u = rho*dot(Constant((0, 0)), v)*dx + p*div(v)*dx

a_p = div(u)*q*dx
L_p = Constant(0)*q*dx

# Calculate cell size manually
h = CellDiameter(mesh)

# Add stabilizations
vms_stab = (1/2)*rho*h**2*dot(grad(div(u)), grad(div(v)))*dx
supg_stab = rho*h*(u.dot(grad(v)) + v.dot(grad(u)))*dx
grad_div_stab = nu*h**2*inner(grad(p), grad(q))*dx

a_u += vms_stab + supg_stab 
a_p += grad_div_stab 

# Solve Navier-Stokes equations
u_ = Function(V)
p_ = Function(Q)
t = 0.0
dt = 0.01
while t < T:
    solve(a_u == L_u, u_, [bc_u])
    solve(a_p == L_p, p_)
    t += dt

# Compute mean drag coefficient
drag_coefficient = assemble((2*nu*u_.dx(0) - p_*u_.dx(0))*ds)

# Save final velocity and pressure fields to file
with XDMFFile('vms_solution.xdmf') as xdmf_file:
    xdmf_file.write(u_, T)
    xdmf_file.write(p_, T)

print("Mean drag coefficient:", drag_coefficient/(0.5*rho*U**2*D))