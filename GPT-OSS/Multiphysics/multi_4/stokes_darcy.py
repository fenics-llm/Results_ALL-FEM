# filename: stokes_darcy.py
from dolfin import *

# --------------------------------------------------------------------
# Parameters (as given in the statement)
# --------------------------------------------------------------------
g     = 1.0
rho   = 1.0
nu    = 1.0
kappa = 1.0          # permeability k
K     = 1.0          # K = k * rho * g / mu  (mu = 1 here)
alpha = 1.0

# --------------------------------------------------------------------
# Geometry and mesh
# --------------------------------------------------------------------
Lx = pi               # length in x‑direction
mesh = RectangleMesh(Point(0.0, -1.0), Point(Lx, 1.0), 64, 64)

# --------------------------------------------------------------------
# Analytical expressions (restricted to the proper sub‑domains)
# --------------------------------------------------------------------
# w(y) and its derivative w'(y)
w_expr  = "-K - (g*x[1])/(2*nu) + (K/2 - alpha*g/(4*nu*nu))*x[1]*x[1]"
dw_expr = "-g/(2*nu) + (K - alpha*g/(2*nu*nu))*x[1]"

# Stokes velocity (only for y>0)
uS = Function(VectorFunctionSpace(mesh, "CG", 2))
uS_expr = Expression((
    "(x[1] > 0 ? (" + dw_expr + ")*cos(x[0]) : 0.0)",
    "(x[1] > 0 ? (" + w_expr + ")*sin(x[0]) : 0.0)"
), degree=4, K=K, g=g, nu=nu, alpha=alpha)
uS.interpolate(uS_expr)

# Darcy pressure (only for y<0)
pD = Function(FunctionSpace(mesh, "CG", 2))
pD_expr = Expression(
    "(x[1] < 0 ? rho*g*exp(x[1])*sin(x[0]) : 0.0)",
    degree=4, rho=rho, g=g)
pD.interpolate(pD_expr)

# --------------------------------------------------------------------
# Save results in XDMF format
# --------------------------------------------------------------------
with XDMFFile(mesh.mpi_comm(), "stokes_velocity.xdmf") as xdmf_u:
    xdmf_u.write(uS)

with XDMFFile(mesh.mpi_comm(), "darcy_pressure.xdmf") as xdmf_p:
    xdmf_p.write(pD)