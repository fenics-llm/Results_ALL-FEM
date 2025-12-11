# filename: q15_fsi.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dolfin import *
from mshr import Rectangle, Circle, generate_mesh

# Geometry
Lx, Ly = 1.20, 0.20
a = 0.03
hole_centers = [(0.30, 0.10), (0.60, 0.10), (0.90, 0.10)]

# Mesh with three holes
domain = Rectangle(Point(0.0, 0.0), Point(Lx, Ly))
for xc, yc in hole_centers:
    domain = domain - Circle(Point(xc, yc), a, 64)
mesh = generate_mesh(domain, 64)

# Material parameters (Pa)
mu = 3.846e6
lmbda = 5.769e6

# Function space for displacement
V = VectorFunctionSpace(mesh, "Lagrange", 2)
u = Function(V, name="Displacement")
du = TrialFunction(V)
v = TestFunction(V)

# Kinematics
I = Identity(2)
F = I + grad(u)
E = 0.5*(F.T*F - I)
J = det(F)
S = lmbda*tr(E)*I + 2.0*mu*E
sigma = (1.0/J)*F*S*F.T

# Weak form
R = inner(sigma, grad(v))*dx
J_form = derivative(R, u, du)

# Boundary conditions
tol = 1E-14
def left(x, on):
    return on and near(x[0], 0.0, tol)
def right(x, on):
    return on and near(x[0], Lx, tol)

load_disp = 0.012
t = Constant(0.0)
u_Rx = Expression("t*load", t=t, load=load_disp, degree=2)

bcs = [DirichletBC(V, Constant((0.0, 0.0)), left),
       DirichletBC(V.sub(0), u_Rx, right),
       DirichletBC(V.sub(1), Constant(0.0), right)]

# Solver parameters
solver_parameters = {"newton_solver":
                     {"relative_tolerance": 1e-6,
                      "absolute_tolerance": 1e-8,
                      "maximum_iterations": 25,
                      "linear_solver": "mumps"}}
set_log_level(LogLevel.ERROR)

# Load stepping
nsteps = 20
max_E_allowed = 0.03
Emax = Function(FunctionSpace(mesh, "DG", 0), name="Emax")

for step in range(1, nsteps+1):
    t.assign(step/float(nsteps))
    solve(R == 0, u, bcs, J=J_form, solver_parameters=solver_parameters)

    # Compute strain tensors for post‑processing
    E_val = 0.5*((I + grad(u)).T*(I + grad(u)) - I)
    Vdg = TensorFunctionSpace(mesh, "DG", 0)
    E_proj = project(E_val, Vdg)
    E11 = E_proj.sub(0,0)
    E22 = E_proj.sub(1,1)
    E12 = E_proj.sub(0,1)

    # Max principal Green–Lagrange strain
    Emax_expr = 0.5*(E11 + E22 + sqrt(pow(E11 - E22, 2) + 4*E12*E12))
    Emax.assign(project(Emax_expr, FunctionSpace(mesh, "DG", 0)))
    Emax_val = Emax.vector().max()
    print("Step {}: t = {:.3f}, max principal strain = {:.5e}".format(step, t(0), Emax_val))

    if Emax_val > max_E_allowed:
        print("Strain limit exceeded – stopping load stepping.")
        break

# Plot deformed configuration
mesh_disp = Mesh(mesh)
u_vert = u.compute_vertex_values().reshape((-1, 2))
mesh_disp.coordinates()[:] += u_vert
plt.figure()
plot(mesh_disp, linewidth=0.5)
plt.title("Deformed configuration")
plt.savefig("q15_def.png", dpi=300)

# Plot max principal strain
plt.figure()
c = plot(Emax, title="Max principal Green–Lagrange strain")
plt.colorbar(c)
plt.savefig("q15_Emax.png", dpi=300)

# Von Mises stress based on second Piola–Kirchhoff stress
s = S - (1.0/3.0)*tr(S)*I
sigma_vm = sqrt(1.5*inner(s, s))
sigma_vm_proj = project(sigma_vm, FunctionSpace(mesh, "DG", 0))
plt.figure()
c = plot(sigma_vm_proj, title="Von Mises stress (S)")
plt.colorbar(c)
plt.savefig("q15_vmS.png", dpi=300)

# Export results
with XDMFFile(mesh.mpi_comm(), "q15_u.xdmf") as xdmf_u:
    xdmf_u.write(u)
with XDMFFile(mesh.mpi_comm(), "q15_Emax.xdmf") as xdmf_E:
    xdmf_E.write(Emax)

print("Simulation finished.")