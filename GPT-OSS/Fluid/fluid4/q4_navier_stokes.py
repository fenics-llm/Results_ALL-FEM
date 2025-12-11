# filename: q4_navier_stokes.py
"""
Steady incompressible Navier–Stokes in a rectangular channel.

Domain: (0, L) × (0, H) with L = 2.0 m, H = 0.20 m.
Mesh: 160 × 16 uniform rectangles.
Boundary conditions:
    - Inlet (x = 0): parabolic profile u_x = 6·Ū·(y/H)·(1‑y/H), u_y = 0,  Ū = 2.5 m/s
    - Walls (y = 0, y = H): no‑slip (u = 0)
    - Outlet (x = L): traction‑free (natural Neumann)
Parameters: μ = 0.01 Pa·s, ρ = 1 kg/m³

Outputs:
    - u_x colour map saved as q4_ux.png
    - Full solution (u, p) saved as q4_soln.xdmf
"""

from dolfin import *
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# Geometry and mesh
L, H = 2.0, 0.20
nx, ny = 160, 16
mesh = RectangleMesh(Point(0.0, 0.0), Point(L, H), nx, ny, "crossed")

# --------------------------------------------------------------
# Function spaces (Taylor‑Hood P2/P1)
V = VectorFunctionSpace(mesh, "Lagrange", 2)   # velocity (P2)
Q = FunctionSpace(mesh, "Lagrange", 1)       # pressure (P1)

# Mixed space V × Q
mixed_elem = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, mixed_elem)

# --------------------------------------------------------------
# Physical parameters
mu = Constant(0.01)          # dynamic viscosity
rho = Constant(1.0)          # density
Ubar = 2.5

# --------------------------------------------------------------
# Inlet velocity expression (parabolic profile)
class InletVelocity(UserExpression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def eval(self, values, x):
        y = x[1]
        values[0] = 6.0 * Ubar * (y / H) * (1.0 - y / H)   # u_x
        values[1] = 0.0                                    # u_y
    def value_shape(self):
        return (2,)

u_inlet = InletVelocity(degree=2)

# --------------------------------------------------------------
# Boundary definitions
class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and on_boundary

class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[1], 0.0) or near(x[1], H)) and on_boundary

class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], L) and on_boundary

inlet = Inlet()
walls = Walls()
outlet = Outlet()

# Dirichlet BCs for velocity (inlet + walls)
bc_inlet = DirichletBC(W.sub(0), u_inlet, inlet)
bc_walls = DirichletBC(W.sub(0), Constant((0.0, 0.0)), walls)

# Pressure reference point to fix the null‑space (point at outlet corner)
class PressurePoint(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], L) and near(x[1], 0.0)

p_point = PressurePoint()
bc_pressure = DirichletBC(W.sub(1), Constant(0.0), p_point, "pointwise")

bcs = [bc_inlet, bc_walls, bc_pressure]

# --------------------------------------------------------------
# Variational formulation (steady Navier–Stokes)
# Unknown and test functions
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

# Current iterate (needed for the nonlinear term)
w = Function(W)               # will hold the solution
(u_k, p_k) = split(w)         # velocity & pressure at current iteration

# Residual (Newton form)
F = (rho * dot(dot(u_k, nabla_grad(u_k)), v) * dx
     + mu * inner(grad(u_k), grad(v)) * dx
     - div(v) * p_k * dx
     + q * div(u_k) * dx)

# Jacobian of the residual
J = derivative(F, w, TrialFunction(W))

# --------------------------------------------------------------
# Solve the nonlinear problem with Newton's method
solve(F == 0, w, bcs, J=J,
      solver_parameters={"newton_solver":
                         {"relative_tolerance": 1e-6,
                          "absolute_tolerance": 1e-8,
                          "maximum_iterations": 30,
                          "linear_solver": "mumps"}})

# Extract velocity and pressure
(u_sol, p_sol) = w.split()

# --------------------------------------------------------------
# Save solution to XDMF
xdmf = XDMFFile(mesh.mpi_comm(), "q4_soln.xdmf")
xdmf.write(u_sol, 0.0)
xdmf.write(p_sol, 0.0)
xdmf.close()

# --------------------------------------------------------------
# Plot u_x component as a colour map and save to PNG
ux = u_sol.sub(0, deepcopy=True)          # x‑component of velocity
ux_vals = ux.compute_vertex_values(mesh)  # values at mesh vertices

plt.figure(figsize=(8, 2))
tpc = plt.tripcolor(mesh.coordinates()[:, 0],
                    mesh.coordinates()[:, 1],
                    mesh.cells(),
                    ux_vals,
                    shading='gouraud',
                    cmap='viridis')
plt.colorbar(tpc, label=r'$u_x$ (m/s)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Velocity component $u_x$')
plt.axis('equal')
plt.tight_layout()
plt.savefig('q4_ux.png', dpi=300)
plt.close()

print("Simulation completed.")
print(" - Velocity & pressure saved to 'q4_soln.xdmf'")
print(" - u_x colour map saved to 'q4_ux.png'")