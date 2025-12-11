from dolfin import *
from mshr import Circle, generate_mesh

# -------------------  Parameters  -------------------
R0    = 5.0e-2          # initial radius (m)
s     = 1.0e-4          # radial growth speed (m/s)
D     = 1.0e-5          # diffusivity (m^2/s)
kappa = 1.0e-4          # decay rate (1/s)
dt    = 0.01            # time step (s)
T     = 10.0            # final time (s)
num_steps = int(T / dt)

# -------------------  Mesh  -------------------
domain = Circle(Point(0.0, 0.0), R0, 64)
mesh   = generate_mesh(domain, int(R0 / 1.0e-3))

# -------------------  Initial condition  -------------------
V = FunctionSpace(mesh, "CG", 1)
c_n = Function(V)
c_n.interpolate(Constant(1.0))

# -------------------  Mesh velocity (ALE)  -------------------
Vv = VectorFunctionSpace(mesh, "CG", 1)
w_expr = Expression(("s*x[0]/(sqrt(x[0]*x[0]+x[1]*x[1])+DOLFIN_EPS)",
                     "s*x[1]/(sqrt(x[0]*x[0]+x[1]*x[1])+DOLFIN_EPS)"),
                    s=s, degree=2)

# -------------------  Output  -------------------
xdmf = XDMFFile(mesh.mpi_comm(), "c.xdmf")
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True

t = 0.0
for n in range(num_steps):
    t += dt

    # --- mesh velocity on current mesh ---
    w = interpolate(w_expr, Vv)

    # --- displacement over one time step ---
    disp = Function(Vv)
    disp.vector()[:] = w.vector() * dt

    # --- move mesh ---
    ALE.move(mesh, disp)

    # --- update function spaces after mesh move ---
    V  = FunctionSpace(mesh, "CG", 1)
    Vv = VectorFunctionSpace(mesh, "CG", 1)

    # --- interpolate previous solution onto new mesh ---
    c_n = interpolate(c_n, V)

    # --- trial / test functions on updated space ---
    c = TrialFunction(V)
    v = TestFunction(V)

    # --- variational forms (Backward Euler) ---
    a = (c*v/dt + D*dot(grad(c), grad(v)) + kappa*c*v + dot(w, grad(v))*c)*dx
    L = (c_n*v/dt)*dx

    # --- solve for new concentration ---
    c = Function(V)
    solve(a == L, c)

    # --- update for next step ---
    c_n.assign(c)

    # --- write solution ---
    xdmf.write(c, t)

    # --- report total mass every 100 steps ---
    if n % 100 == 0:
        mass = assemble(c*dx)
        print("t = %.2f s, total mass = %.6g" % (t, mass))

xdmf.close()