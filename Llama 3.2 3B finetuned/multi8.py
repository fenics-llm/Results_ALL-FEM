from dolfin import *
import mshr
import matplotlib.pyplot as plt

# Geometry
domain = mshr.Rectangle(Point(0, 0), Point(1, H)) - \
         mshr.Rectangle(Point(0.6, 0), Point(1, H))
mesh_f = mshr.generate_mesh(domain, 64)
mesh_p = mshr.generate_mesh(mshr.Rectangle(Point(0.6, 0), Point(1, H)), 64)

# Meshes
mesh = Mesh(mesh_f + mesh_p)

# Function space
V = VectorFunctionSpace(mesh, "P", 1)

# Mark subdomains
class Fluid(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.6)
class Porous(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and not near(x[0], 0.6)

fluid = Fluid()
porous = Porous()

# Mark boundary
bdr = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
bdr.set_all(0)
fluid.mark(bdr, 1)
porous.mark(bdr, 2)
ds = Measure("ds", domain=mesh, subdomain_data=bdr)

# Velocity and pressure in Fluid
u_f, p_f = TrialFunctions(V), TestFunctions(V)
a_f = (dot(grad(u_f), grad(v)) - div(v)*p_f) * dx
L_f = Constant(0.0)*v*dx

# Velocity and pressure in Porous
u_p, p_p = TrialFunctions(V), TestFunctions(V)
a_p = ((K/μ)*(grad(p_p)-grad(p_f))*grad(v) + dot(u_p, grad(v)))*dx
L_p = Constant(0.0)*v*dx

# Interface conditions
tangential = project(dot(u_f - u_p, x[1]*x[2] - y*x[3]), V)
a += a_f + a_p
L += L_f + L_p
bc_tang = DirichletBC(V, tangential, fluid)

# External inlet velocity
u_in = Expression(("6*U_bar*y*(H-y)/H^2", "0.0"), U_bar=1e-2, H=0.2)
bcs.append(bc_tang)

# Fluid walls: no slip and no penetration
noslip = Constant((0.0, 0.0))
bc_wall_f = DirichletBC(V, noslip, bdr, 1)
bcs.append(bc_wall_f)

# Porous walls: no-flux (impermeable)
noslip_p = Constant((0.0, 0.0))
bc_wall_p = DirichletBC(V, noslip_p, bdr, 2)
bcs.append(bc_wall_p)

# Outlet
p_out = Constant(0.0)
bc_out = DirichletBC(V, p_out, bdr, 3)
bcs.append(bc_out)

# Solve
u, p = Function(V), Function(V)
solve(a == L, u, bcs=bcs)
p_f.assign(p.sub(0))
p_p.assign(p.sub(1))

# Plotting
plt.figure()
plot(u, title="Velocity (m/s)")
plt.colorbar()
plt.figure()
plot(p, title="Pressure (Pa)")
plt.savefig("q15_solution.png")

# Interface profiles
u_x = project(dot(u, x[1]*x[2] - y*x[3]), V)
uy = project(dot(u, x[0]*y - z*x[1]), V)

with open("q15_interface.csv", "w") as f:
    f.write("y,ux,uy\n")
    for y in np.linspace(0, H, 101):
        f.write(f"{y:.4f},{u_x(y):.6f},{uy(y):.6f}\n")