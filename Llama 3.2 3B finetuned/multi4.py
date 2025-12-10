from dolfin import *
import mshr
import matplotlib.pyplot as plt

# Geometry
domain = mshr.Rectangle(Point(0, 0), Point(pi, 1)) - mshr.Rectangle(Point(0, 0), Point(pi, -1))
mesh = generate_mesh(domain, 64)

# Stokes region: upper half of rectangle (x=0..π)
Omega_S = Rectangle(Point(0, 0), Point(pi, 1))

# Darcy region: lower half of rectangle (x=0..π)
Omega_D = Rectangle(Point(0, -1), Point(pi, 0))

# Stokes velocity
u_S = TrialFunction(V)
v_S = TestFunction(V)

# Stokes pressure
p_S = TrialFunction(W)
q_S = TestFunction(W)

# Darcy velocity
u_D = TrialFunction(U)
v_D = TestFunction(U)

# Darcy pressure
p_D = TrialFunction(P)

# Stokes body force
b_x = Expression("((ν*K - (α*g)/(2*ν))*y - g/2)*cos(x)", ν=1.0, K=1.0, α=1.0, g=1.0)
b_y = Expression(
    "[( (ν*K)/2 - (α*g)/(4*ν) )*y^2 - (g/2)*y + ((α*g)/(2*ν) - 2*ν*K)]*sin(x)",
    ν=1.0,
    K=1.0,
    α=1.0,
    g=1.0
)

# Stokes variational forms
a_S = inner(grad(u_S), grad(v_S)) * dx
L_S = dot(b_x, v_S) * dx + dot(b_y, v_S) * ds

# Darcy variational form
a_D = inner(u_D, grad(v_D)) * dx
L_D = 0.0

# Stokes–Darcy interface conditions
t = Expression("1.0", degree=2)
n = Expression("0.0", degree=2)

# Stokes–Darcy mass continuity
a_interface = dot(u_S - u_D, v_D) * ds
L_interface = 0.0

# Stokes–Darcy normal traction balance
a_n = inner(n * sigma(S), n * q_S) * ds
L_n = -dot(p_D / rho, q_S) * ds

# Stokes–Darcy tangential condition
a_t = dot(t * (sigma(S) - sigma(D)), v_D) * ds
L_t = 0.0

# Stokes variational form with interface conditions
a_total = a_S + a_interface + a_n + a_t
L_total = L_S + L_interface + L_n + L_t

# Assemble and solve
A = assemble(a_total)
b = assemble(L_total)

u_S_sol = Function(V)
p_S_sol = Function(W)
u_D_sol = Function(U)
p_D_sol = Function(P)

solve(A, u_S_sol.vector(), b)
solve(A, p_S_sol.vector(), b)
solve(A, u_D_sol.vector(), b)
solve(A, p_D_sol.vector(), b)

# Save Stokes velocity and Darcy pressure in XDMF format
u_S_sol_file = File("stokes_velocity.xdmf")
p_S_sol_file = File("stokes_pressure.xdmf")

u_S_sol_file << u_S_sol
p_S_sol_file << p_S_sol

u_D_sol_file = File("darcy_velocity.xdmf")
p_D_sol_file = File("darcy_pressure.xdmf")

u_D_sol_file << u_D_sol
p_D_sol_file << p_D_sol

plt.figure()
plot(u_S_sol, title="Stokes velocity (x=0)")
plt.figure()
plot(p_S_sol, title="Stokes pressure (x=0)")
plt.show()
