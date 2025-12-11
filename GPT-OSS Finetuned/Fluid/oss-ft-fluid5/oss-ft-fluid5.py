# filename: q5_navier_stokes.sh
#!/usr/bin/env bash
# Write the FEniCS Python script
#!/usr/bin/env python3
import dolfin as dl
import matplotlib.pyplot as plt

# Parameters
rho = 1.0
mu = 0.01
U_lid = 1.0

# Mesh
nx = ny = 128
mesh = dl.UnitSquareMesh(nx, ny)

# Mixed Taylor–Hood element (P2 for velocity, P1 for pressure)
V_el = dl.VectorElement("Lagrange", mesh.ufl_cell(), 2)
Q_el = dl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
W_el = dl.MixedElement([V_el, Q_el])
W = dl.FunctionSpace(mesh, W_el)

# Boundary definitions
tol = 1e-14
class Lid(dl.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and dl.near(x[1], 1.0, tol)
class Walls(dl.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (dl.near(x[1], 0.0, tol) or dl.near(x[0], 0.0, tol) or dl.near(x[0], 1.0, tol))
class PressurePoint(dl.SubDomain):
    def inside(self, x, on_boundary):
        return dl.near(x[0], 0.0, tol) and dl.near(x[1], 0.0, tol)

lid = Lid()
walls = Walls()
p_point = PressurePoint()

# No-slip on walls
noslip = dl.Constant((0.0, 0.0))
bc_walls = dl.DirichletBC(W.sub(0), noslip, walls)

# Lid velocity
lid_vel = dl.Constant((U_lid, 0.0))
bc_lid = dl.DirichletBC(W.sub(0), lid_vel, lid)

# Pressure gauge at (0,0) – pointwise Dirichlet
bc_p = dl.DirichletBC(W.sub(1), dl.Constant(0.0), p_point, method="pointwise")

bcs = [bc_walls, bc_lid, bc_p]

# Unknown mixed function
w = dl.Function(W)               # (u,p)
(u, p) = dl.split(w)            # velocity, pressure
(v, q) = dl.TestFunctions(W)

# Strain-rate tensor and stress
def epsilon(u):
    return dl.sym(dl.grad(u))
def sigma(u, p):
    return 2*mu*epsilon(u) - p*dl.Identity(len(u))

# Steady Navier–Stokes variational form
F = (rho*dl.dot(dl.grad(u)*u, v) + dl.inner(sigma(u, p), epsilon(v)) - dl.div(u)*q)*dl.dx

# Jacobian
J = dl.derivative(F, w, dl.TrialFunction(W))

# Solve nonlinear problem
dl.solve(F == 0, w, bcs, J=J,
          solver_parameters={"newton_solver":
                              {"relative_tolerance": 1e-8,
                               "absolute_tolerance": 1e-10,
                               "maximum_iterations": 25,
                               "linear_solver": "mumps"}})

# Extract velocity and pressure on collapsed subspaces
u_sol, p_sol = w.split(deepcopy=True)

# Compute speed magnitude
speed = dl.sqrt(dl.dot(u_sol, u_sol))

# Plot speed and save as PNG
plt.figure(figsize=(6,5))
c = dl.plot(speed, title="Speed |u|", cmap="viridis")
plt.colorbar(c)
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.savefig("q5_speed.png", dpi=300)
plt.close()

# Save solution to XDMF
with dl.XDMFFile(mesh.mpi_comm(), "q5_soln.xdmf") as xdmf:
    xdmf.write(u_sol)
    xdmf.write(p_sol)