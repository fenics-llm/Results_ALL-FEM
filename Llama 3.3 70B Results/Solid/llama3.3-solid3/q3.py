from fenics import *
import numpy as np
import matplotlib.pyplot as plt

# Create mesh
mesh = RectangleMesh(Point(0, 0), Point(1.0, 0.20), 100, 20)

# Define the circular hole
class Hole(SubDomain):
    def inside(self, x, on_boundary):
        return (x[0] - 0.50)**2 + (x[1] - 0.10)**2 < 0.05**2

hole = Hole()
sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
hole.mark(sub_domains, 1)

# Create a new mesh with the hole
mesh_with_hole = SubMesh(mesh, sub_domains, 0)

# Define material parameters
E = 200e3  # Young's modulus in MPa
nu = 0.30  # Poisson's ratio

# Create function space
V = VectorFunctionSpace(mesh_with_hole, 'P', 2)

# Define boundary conditions
u_L = Constant((0, 0))
def left_boundary(x, on_boundary):
    return near(x[0], 0) and on_boundary

bc_L = DirichletBC(V, u_L, left_boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)

# Define stress and strain tensors
def sigma(u, E, nu):
    mu = E / (2 * (1 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    return 2 * mu * sym(grad(u)) + lmbda * tr(sym(grad(u))) * Identity(2)

def epsilon(u):
    return sym(grad(u))

# Define the traction on the right boundary
class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 1.0) and on_boundary

right_boundary = RightBoundary()
sub_domains_right = MeshFunction("size_t", mesh_with_hole, mesh_with_hole.topology().dim(), 0)
right_boundary.mark(sub_domains_right, 2)

ds_right = Measure('ds', subdomain_data=sub_domains_right)(subdomain_id=2)

# Define variational problem with traction
a = inner(sigma(u, E, nu), epsilon(v)) * dx
L = dot(Constant((2e6, 0)), v) * ds_right

# Assemble and solve the system
u_sol = Function(V)
solve(a == L, u_sol, [bc_L])

# Compute von Mises stress
def von_mises(stress):
    return sqrt(3*stress[0,1]**2 + 3*stress[1,0]**2 + (stress[0,0]-stress[1,1])**2 + stress[0,0]**2 + stress[1,1]**2)

vm_stress = project(von_mises(sigma(u_sol, E, nu)), FunctionSpace(mesh_with_hole, 'P', 1))

# Save color map
plt.figure()
p = plot(vm_stress)
plt.colorbar(p)
plt.savefig('q3_vm.png')

# Save displacement field in XDMF format
u_sol.rename("Displacement", "displacement")
file = XDMFFile("displacement.xdmf")
file.write(u_sol)

# Report maximum von Mises stress at the hole boundary and Stress Concentration Factor (Kt)
max_vm_stress = vm_stress.compute_vertex_values(mesh_with_hole)
hole_boundary_ids = np.where(sub_domains.array() == 1)[0]
valid_ids = [i for i in hole_boundary_ids if i < len(max_vm_stress)]
if valid_ids:
    max_vm_stress_at_hole_boundary = max([max_vm_stress[i] for i in valid_ids])
    Kt = max_vm_stress_at_hole_boundary / (2e6)
    print(f"Maximum von Mises stress at the hole boundary: {max_vm_stress_at_hole_boundary} Pa")
    print(f"Stress Concentration Factor (Kt): {Kt}")
else:
    print("No valid indices found for hole boundary.")