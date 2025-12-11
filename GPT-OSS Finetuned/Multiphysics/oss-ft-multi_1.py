# filename: expanding_disk_ale.py
from dolfin import *
import mshr
import numpy as np

# -------------------  Parameters  -------------------
R0    = 0.05          # initial radius (m)
s     = 1.0e-4        # radial growth speed (m/s)
D     = 1.0e-5        # diffusivity (m^2/s)
kappa = 1.0e-4        # decay rate (1/s)
dt    = 0.01          # time step (s)
T     = 10.0          # final time (s)
h0    = 1.0e-3        # target mesh size (m)

# -------------------  Mesh generation  -------------------
domain = mshr.Circle(Point(0.0, 0.0), R0, int(2*np.pi*R0/h0))
mesh   = mshr.generate_mesh(domain, int(2*np.pi*R0/h0))

# -------------------  Function space  -------------------
V = FunctionSpace(mesh, "Lagrange", 1)

# -------------------  Initial condition  -------------------
c_n = interpolate(Constant(1.0), V)   # c at previous step

# -------------------  XDMF output  -------------------
xdmf = XDMFFile(mesh.mpi_comm(), "c_expanding_disk.xdmf")
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True

# -------------------  Time stepping  -------------------
t = 0.0
step = 0
while t < T + DOLFIN_EPS:
    # ---- mesh velocity (computed on current mesh) ----
    Vvec = VectorFunctionSpace(mesh, "Lagrange", 1)
    w = Function(Vvec)
    w_vec = w.vector()
    x = mesh.coordinates()
    r = np.sqrt(x[:,0]**2 + x[:,1]**2)
    disp = np.zeros_like(x)
    mask = r > DOLFIN_EPS
    disp[mask,0] = s*dt * x[mask,0] / r[mask]
    disp[mask,1] = s*dt * x[mask,1] / r[mask]
    w_vec.set_local(disp.ravel())

    # ---- move mesh ----
    ALE.move(mesh, w)

    # ---- update function space after mesh move ----
    V = FunctionSpace(mesh, "Lagrange", 1)
    c_n = interpolate(c_n, V)   # project previous solution onto new mesh

    # ---- variational problem (backward Euler, ALE) ----
    c = TrialFunction(V)
    v = TestFunction(V)

    a = (c*v + dt*D*dot(grad(c), grad(v)) + dt*kappa*c*v)*dx
    L = (c_n*v - dt*dot(w, grad(c_n))*v)*dx

    c = Function(V)
    solve(a == L, c)

    # ---- output ----
    if step % 100 == 0:
        total = assemble(c*dx)
        print("t = {:.2f} s, total mass = {:.6e}".format(t, total))
        xdmf.write(c, t)

    # ---- advance time ----
    t += dt
    step += 1

xdmf.close()
print("Simulation finished.")