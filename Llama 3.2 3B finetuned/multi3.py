from dolfin import *
import mshr
import matplotlib.pyplot as plt

# Parameters
theta = 1.5
alpha = 3000.
T = 4e-2

# Mesh and function space
mesh = mshr.RectangleMesh(mshr.Rectangle(Point(0, 0), Point(1, 1)), 64, 64)
V = FunctionSpace(mesh, "CG", 1)

# Initial condition
c_bar = Constant(0.63)
r = Expression("0.05 * (x[0] + x[1]) / 2.0", degree=2)
u0 = c_bar + r

# Time-stepping parameters
dt = 5e-7
t = 0.
n_steps = int(T/dt)

# Mobility function
M_c = Expression("c*(1-c)", c=Constant(0.5), degree=2)

# Function for chemical potential μ
mu_c = Expression(
    "0.5 * log(c/(1 - c)) + (1 - 2*c)",
    c=c_bar,
    degree=4,
)
mu = Constant(3*alpha*mu_c)

# Time-stepping loop
for n in range(n_steps):
    t += dt

    # Compute time step size
    if t > T:
        dt = T - t
    elif t < 1e-6:
        dt = 5e-7
    else:
        dt = min(dt*0.9, 5e-7)

    # Solve the Cahn–Hilliard equation
    c_n = Function(V)
    u_n = Function(V)
    b = Constant(0.)
    for i in range(10):
        solve(
            (dot(grad(c_n), grad(v)) + alpha*dot(M_c, grad(mu))) * v == b * v,
            v,
            c_n + dt*u_n,
            mesh,
            "cg",
        )
        u = -M_c * grad(mu)
        solve((dot(grad(u_n), grad(w)) + dot(M_c, grad(mu))) * w == b * w, w, u, mesh, "cg")
    b = Constant(0.)
    for i in range(10):
        solve(
            (dot(grad(c_n), grad(v)) + alpha*dot(M_c, grad(mu))) * v == b * v,
            v,
            c_n + dt*u_n,
            mesh,
            "cg",
        )
        u = -M_c * grad(mu)
        solve((dot(grad(u_n), grad(w)) + dot(M_c, grad(mu))) * w == b * w, w, u, mesh, "cg")

    # Save the solution
    file = XDMFFile("cahn_hilliard.xdmf")
    file.parameters["flush_output"] = True
    file.parameters["functions_share_mesh"] = True
    file.write(c_n, t)
    file.write(u, t)