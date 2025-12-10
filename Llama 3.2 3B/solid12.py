# filename: femics.py

from fenics import *

# Create the mesh
mesh = UnitSquareMesh(100, 20)

# Define the boundary conditions
def left_boundary(x):
    return (0, 0)

def right_boundary(x):
    return (1, 0)

def top_boundary(x):
    return (0.5, 1)

def bottom_boundary(x):
    return (0.5, 0)

def hole_boundary(x):
    return (0.5 + 0.04 * sin(2 * pi * x[0]), 0.1)

# Create the boundary conditions
bc_left = DirichletBC(FunctionSpace(mesh.subdomain(0), 'P'), Constant(0), left_boundary)
bc_right = DirichletBC(FunctionSpace(mesh.subdomain(1), 'P'), Constant(0.06), right_boundary)
bc_top = DirichletBC(FunctionSpace(mesh.subdomain(2), 'P'), Constant(0.1), top_boundary)
bc_bottom = DirichletBC(FunctionSpace(mesh.subdomain(3), 'P'), Constant(0), bottom_boundary)
bc_hole = DirichletBC(FunctionSpace(mesh.subdomain(4), 'P'), Constant(0), hole_boundary)

# Define the material properties
E = 50000000  # Pa
nu = 0.5

# Create the function spaces
V = FunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

# Define the displacement and pressure fields
u = TrialFunction(V)
p = TrialFunction(Q)

# Compute the strain energy
W_u = 0.5 * integrate(nu * E * inner(grad(u), grad(u)) + (1 - 2 * nu) * E * inner(grad(u), grad(u)))
W_p = 0.5 * E * integrate(inner(p, p))

# Define the Lagrange multiplier for incompressibility
phi = TrialFunction(Q)

# Compute the variational form of the problem
F_u = W_u + (1 / E) * integrate(W_p)
F_p = W_p - (1 / E) * integrate(inner(grad(u), grad(p)) - inner(grad(phi), grad(phi)))
F_phi = 0.5 * E * integrate(inner(phi, phi))

# Define the boundary conditions
bc_u = F_u.subs({u: u.lhs, p: p.lhs})
bc_p = F_p.subs({p: p.lhs, phi: phi.lhs})

# Assemble the variational form
F = bc_u + bc_p + F_phi

# Compute the solution
u_solution = Function(V)
p_solution = Function(Q)
phi_solution = Function(Q)

solve(F, u_solution, p_solution, phi_solution, bc_u, bc_p, bc_phi=True)

# Save the displacement field in XDMF format
xdmf_file = File("displacement.xdmf")
xdmf_file << u_solution

# Compute and save the color map of hydrostatic pressure p as q12_p.png.
q12_p = p_solution
plot(q12_p)
savefig("q12_p.png")

# Compute and save the color map of von Mises stress (from the Cauchy stress) as q12_vm.png.
sigma = 2 * E * inner(grad(u), grad(u)) - E * p * I
vm = sqrt(3 / 2) * norm(sigma)
q12_vm = vm
plot(q12_vm)
savefig("q12_vm.png")