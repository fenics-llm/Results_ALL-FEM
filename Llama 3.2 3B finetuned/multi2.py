# filename: phase-field.py

from dolfin import *
import mshr
import matplotlib.pyplot as plt

# Define the unit square domain
domain = mshr.Rectangle(Point(0, 0), Point(1, 1))
mesh = mshr.generate_mesh(domain, 200)

# Define the potential function W
W = Expression("0.25*(pow(x[0],2)-1)*pow(x[1],2)", degree=4)
M = Constant(1.0) # mobility

# Define the phase field ϕ
V = FunctionSpace(mesh, "P", 1)
u = TrialFunction(V)
v = TestFunction(V)

# Variational form
a = M * dot(grad(u), grad(v)) * dx
L = Constant(0.0) * v * dx

# Time-stepping loop
t = 0.0
T = 0.20
dt = 1e-3
n = 5 # number of time steps
for i in range(n):
    t += dt
    
    # Compute the signed distance function d_rect
    d_rect = Expression("sqrt(pow(x[0]-0.5,2)+pow(x[1]-0.5,2))", degree=4)
    
    # Define the phase field ϕ at time t
    u_expr = Expression("tanh(d_rect/(sqrt(2)*%f))" % (epsilon), epsilon=0.01, degree=3)
    u_sol = interpolate(u_expr, V)

    # Compute the curvature term W′(ϕ) and the Laplacian ∇²ϕ
    w_prime = derivative(W, u_sol)
    
    # Assemble the variational form
    A = assemble(a)
    b = assemble(L)
    
    # Solve for ϕ at time t + dt
    solve(A, u.vector(), b)

# Plot the phase field at each time step
plt.figure()
for i in range(n):
    plt.subplot(2, 3, i+1)
    plot(u_sol, title="Phase field at t = %g" % (t))
    plt.colorbar()
plt.show()