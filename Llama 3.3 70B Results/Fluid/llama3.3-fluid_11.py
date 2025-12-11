from fenics import *

# Define constants
L = 2.0         # length
H = 0.20        # height
rho = 1.0       # density
mu0 = 0.5       # consistency index
n = 0.5         # flow behavior index
U_bar = 1.0     # average velocity

# Create mesh
mesh = RectangleMesh(Point(0, 0), Point(L, H), 240, 24)

# Define function spaces
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

# Define stress tensor and effective viscosity
def sigma(u):
    D = sym(grad(u))
    return 2 * mu0 * (inner(D, D))**((n-1)/2) * D

# Define variational problem
F = rho * inner(dot(u, grad(u)), v)*dx + inner(sigma(u), grad(v))*dx - inner(p, div(v))*dx + inner(div(u), q)*dx

# Define boundary conditions
u_inlet = Expression('6 * U_bar * x[1] * (H - x[1]) / H/H', U_bar=U_bar, H=H, degree=2)
bcs_u = [DirichletBC(V.sub(0), u_inlet, 'on(x[0]==0)'),
         DirichletBC(V.sub(1), 0, 'on(x[0]==0)'),
         DirichletBC(V, (0, 0), 'on(x[1]==0 || x[1]==H)')]

# Solve variational problem
u_sol = Function(V)
p_sol = Function(Q)
solve(F == 0, [u_sol, p_sol], bcs=bcs_u)

# Compute effective viscosity
mu_eff = mu0 * (2 * inner(sym(grad(u_sol)), sym(grad(u_sol))))**((n-1)/2)

# Save velocity magnitude to file
import numpy as np
u_mag = np.sqrt(u_sol.sub(0)**2 + u_sol.sub(1)**2)
vtkfile = File('q12_speed.pvd')
vtkfile << u_mag

# Extract streamwise velocity profile at mid-channel
ux_mid = []
y_values = np.linspace(0, H, 100)
for y in y_values:
    ux_mid.append(u_sol.sub(0).at(Point(1.0, y)))

import csv
with open('q12_profile.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['y', 'ux'])
    for i in range(len(y_values)):
        writer.writerow([y_values[i], ux_mid[i]])

# Save solution to file
xdmffile = XDMFFile('q12_solution.xdmf')
xdmffile.write(u_sol, 0)
xdmffile.write(p_sol, 0)
xdmffile.write(mu_eff, 0)

# Report maximum velocity at x = L/2
max_ux = max(ux_mid)
print(max_ux)