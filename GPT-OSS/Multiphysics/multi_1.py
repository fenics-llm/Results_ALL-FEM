# filename: transport_expanding_disk.py
"""
Transport of a chemical in an expanding circular disk using ALE (FEniCS).

Domain:   Ω(t) = { x ∈ R^2 : |x| < R(t) },   R(t) = R0 + s·t
Mesh:     Unstructured triangular mesh of Ω(0) (h0 ≈ 1e-3 m)
PDE:      ∂c/∂t + w·∇c - ∇·(D∇c) + κ c = 0   in Ω(t)
Flux BC:  (-D∇c - w c)·n = 0                on Γ(t)
IC:       c(x,0) = 1
Parameters:
    R0 = 0.05          # m
    s  = 1.0e-4        # m/s  (radial mesh velocity magnitude)
    D  = 1.0e-5        # m^2/s
    κ  = 1.0e-4        # 1/s
    dt = 0.01          # s
    T  = 10.0          # s
"""

from dolfin import *
import mshr
import numpy as np

# -------------------- parameters --------------------
R0 = 0.05
s  = 1.0e-4
D  = 1.0e-5
kappa = 1.0e-4
dt = 0.01
T  = 10.0
num_steps = int(T / dt)

# -------------------- initial mesh --------------------
circle = mshr.Circle(Point(0.0, 0.0), R0)
# Estimate number of cells to obtain h≈1e-3
mesh = mshr.generate_mesh(circle, int(2 * np.pi * R0 / 1e-3))

# Function space (will be recreated after each ALE move)
V = FunctionSpace(mesh, "CG", 1)

# Initial condition c(x,0)=1
c_n = interpolate(Constant(1.0), V)   # Function holding solution at previous step

# Helper to compute mesh velocity w = s * x / |x|
def mesh_velocity(mesh):
    X = SpatialCoordinate(mesh)
    r = sqrt(X[0]**2 + X[1]**2 + DOLFIN_EPS)
    return as_vector((s * X[0] / r, s * X[1] / r))

# Output file
xdmf_file = XDMFFile(mesh.mpi_comm(), "concentration.xdmf")
xdmf_file.parameters["flush_output"] = True
xdmf_file.parameters["functions_share_mesh"] = True
xdmf_file.write(c_n, 0.0)

# -------------------- time stepping --------------------
t = 0.0
for n in range(1, num_steps + 1):
    t += dt

    # ---- move mesh (ALE) ----
    w_cur = mesh_velocity(mesh)
    displacement = project(w_cur * dt, VectorFunctionSpace(mesh, "CG", 1))
    ALE.move(mesh, displacement)

    # Re‑create function space on the new mesh
    V = FunctionSpace(mesh, "CG", 1)

    # Interpolate previous solution onto the new mesh
    c_n = interpolate(c_n, V)

    # Define trial / test functions
    u = TrialFunction(V)
    v = TestFunction(V)

    # Mesh velocity on the new mesh
    w = mesh_velocity(mesh)

    # Variational formulation (backward Euler)
    a = (u / dt) * v * dx + dot(w, grad(u)) * v * dx + D * dot(grad(u), grad(v)) * dx + kappa * u * v * dx
    L = (c_n / dt) * v * dx

    # Assemble and solve
    A = assemble(a)
    b = assemble(L)
    c = Function(V)               # solution at new time level
    solve(A, c.vector(), b, "cg", "ilu")

    # Update previous solution
    c_n.assign(c)

    # Write solution to XDMF (every step)
    xdmf_file.write(c, t)

    # Report total concentration every 100 steps
    if n % 100 == 0:
        total_mass = assemble(c * dx)
        print(f"t = {t:5.2f} s, total concentration = {total_mass:.6e}")

xdmf_file.close()
print("Simulation finished.")