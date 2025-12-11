# filename: problem8.py

from dolfin import *
import matplotlib.pyplot as plt
import mshr

# Geometry and mesh
Lx, Ly = 1.0, 0.20
domain = mshr.Rectangle(Point(0, 0), Point(Lx, Ly))
mesh = mshr.generate_mesh(domain, 50)

# Function space
V = VectorFunctionSpace(mesh, "P", 1)
u = TrialFunction(V)
v = TestFunction(V)

# Material properties (orthotropic lamina in local axes)
E1, E2, G12, nu12 = 40e9, 10e9, 5e9, 0.25
Q11, Q22, Q66, Q12, Q26, Q16 = E1/(1 - nu12**2), E2/(1 - nu12**2), G12, \
    nu12*E2/(1 - nu12**2), 0.0, 0.0

# Rotation angle (30 degrees)
theta = pi/6
c, s = cos(theta), sin(theta)

# Stiffness matrix in global axes via rotation
Qxx = c**4*Q11 + 2*c**2*s**2*(Q12+2*Q66) + s**4*Q22
Qyy = s**4*Q11 + 2*c**2*s**2*(Q12+2*Q66) + c**4*Q22
Qxy = c**2*s**2*(Q11+Q22-2*Q12) + (c**2-s**2)**2*Q66
Qyx = Qxy

# Stiffness matrix in global axes via rotation
Q = as_matrix([[Qxx, Qxy, 0.0], [Qyx, Qyy, 0.0], [0.0, 0.0, 2*G12]])

# Boundary markers
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0) and on_boundary

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], Ly) and on_boundary

Bottom().mark(boundaries, 1)
Top().mark(boundaries, 2)

ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

# Boundary conditions
bc_bottom = DirichletBC(V, Constant((0.0, 0.0)), boundaries, 1)

# Traction on top edge (σ n = (0, -10 MPa))
t_top = Constant((0.0, -10e6))

# Variational problem
def epsilon(u):
    return as_tensor([[u[0].dx(0), u[0].dx(1)],
                      [u[1].dx(0), u[1].dx(1)]])
def sigma(u):
    return as_tensor([[Q[0, 0]*epsilon(u)[0, 0] + Q[0, 1]*epsilon(u)[1, 1],
                       Q[0, 2]*(epsilon(u)[0, 1] + epsilon(u)[1, 0])],
                      [Q[1, 0]*epsilon(u)[0, 0] + Q[1, 1]*epsilon(u)[1, 1],
                       Q[1, 2]*(epsilon(u)[0, 1] + epsilon(u)[1, 0])]])

a = inner(sigma(u), epsilon(v))*dx
L = dot(t_top, v)*ds(2)

# Solve
u_sol = Function(V)
solve(a == L, u_sol, bc_bottom)

# Postprocessing: von Mises stress
sxx, syy, sxy = sigma(u_sol)[0, 0], sigma(u_sol)[1, 1], sigma(u_sol)[0, 1]
vm = project(sqrt(0.5*((sxx-syy)**2 + 3*sxy**2)), FunctionSpace(mesh, "P", 1))

# Visualization
ux_mag = project(u_sol[0], FunctionSpace(mesh, "P", 1))
plt.figure(figsize=(10, 4))
plt.subplot(121)
p = plot(ux_mag, title="Horizontal displacement (m)")
plt.colorbar(p)
plt.savefig("q8_ux.png")
plt.subplot(122)
p = plot(vm, title="von Mises stress (Pa)")
plt.colorbar(p)
plt.savefig("q8_vm.png")

# Save to XDMF
file = XDMFFile("q8_solution.xdmf")
file.parameters["flush_output"] = True
file.parameters["functions_share_mesh"] = True
file.write(u_sol, 0.0)
file.write(vm, 0.0)