from dolfin import *

# Geometry and mesh
mesh = UnitSquareMesh(100, 100)

# Function spaces
V = FunctionSpace(mesh, "CG", 1)
ME = MixedElement([V.ufl_element(), V.ufl_element()])
W = FunctionSpace(mesh, ME)

class PeriodicBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool(on_boundary and (near(x[0], 0) or near(x[1], 0)))

    def map(self, x, y):
        if near(x[0], 1) and near(x[1], 1):
            y[0] = x[0] - 1.0
            y[1] = x[1] - 1.0
        elif near(x[0], 1):
            y[0] = x[0] - 1.0
            y[1] = x[1]
        else:
            y[0] = x[0]
            y[1] = x[1] - 1.0

# Periodic boundary conditions
bc = PeriodicBoundary()

# Trial and test functions
du    = TrialFunction(W)
q, v  = TestFunctions(W)

# Mesh functions
u     = Function(W)
u_1   = Function(W)

# Split mixed functions
dc, dmu = split(du)
c, mu  = split(u)
c_old, _ = split(u_1)

# Parameters
theta    = 1.5
alpha    = 3000.0
dt       = 3e-7

# Initial condition
c_avg   = Constant(0.63)
r       = Expression("0.05*(rand() - 0.5)", degree=2)
u_init  = project(c_avg + r, V)

# Assign initial values
assign(u_1.sub(0), u_init)
assign(u.sub(0), u_init)

# Output file
file = XDMFFile("cahn_hilliard.xdmf")
file.parameters["flush_output"] = True

# Time stepping parameters
t  = 0.0
T  = 0.04
count = 0

while t < T + DOLFIN_EPS:
    # Compute chemical potential df/dc
    f    = (0.5/theta)*ln(c/(1-c)) + c - 2*c**2
    dfdc = diff(f, c)

    # Degenerate mobility function
    M = c*(1-c)

    # Weak statement of the equations
    L0 = (c - c_old)/dt*q*dx - M*dot(grad(dmu), grad(q))*dx
    L1 = dmu*v*dx + dfdc*v*c*dx - 2*alpha*dot(grad(c), grad(v))*dx

    a = lhs(L0 + L1)
    L = rhs(L0 + L1)

    # Solve tentative step
    solve(a == L, u)

    # Check nonlinear convergence
    c_new, _ = u.split()
    diff = errornorm(c_old, c_new, "L2")
    print("Time step: %e,  Diff: %.4f" % (dt, diff))

    if diff < 1e-10:
        dt *= 2.0

    # Check divergence and reduce time step
    if diff > 5e-3 or count > 8:
        assign(u_1.sub(0), c_old)
        dt /= 2.0
        print("Reducing time step to: ", dt, " Diff: %.4f" % diff)
        count = 0

    # Update c and t
    assign(c_old, u.split()[0])
    t += dt
    count += 1

    # Output solution
    if abs(t - 3e-6) < DOLFIN_EPS or \
       abs(t - 1e-4) < DOLFIN_EPS or \
       abs(t - 1e-3) < DOLFIN_EPS or \
       abs(t - 4e-2) < DOLFIN_EPS:
        file.write(c, t)