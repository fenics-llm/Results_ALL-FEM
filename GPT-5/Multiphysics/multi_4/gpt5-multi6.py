from dolfin import *
import math

# --------------------------------------------------------------------
# Parameters (all set to 1 as requested)
# --------------------------------------------------------------------
g = 1.0
rho = 1.0
nu = 1.0
k = 1.0
alpha = 1.0
# From K = k * rho * g / mu  and K = 1  ⇒  mu = k * rho * g / K = 1
K = 1.0
mu = 1.0

# --------------------------------------------------------------------
# Geometry: Ω = (0, π) × (−1, 1)
# Subdomains: Ω_S (y in [0,1]), Ω_D (y in [-1,0])
# --------------------------------------------------------------------
nx, ny = 128, 128  # refine as needed
mesh = RectangleMesh(Point(0.0, -1.0), Point(math.pi, 1.0), nx, ny)

# Mark cells: 1 = Stokes (y >= 0), 2 = Darcy (y <= 0)
subdomains = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)

class OmegaS(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] >= -DOLFIN_EPS and x[1] <= 1.0 + DOLFIN_EPS

class OmegaD(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] <= DOLFIN_EPS and x[1] >= -1.0 - DOLFIN_EPS

OmegaS().mark(subdomains, 1)
OmegaD().mark(subdomains, 2)

# Submeshes
submesh_S = SubMesh(mesh, subdomains, 1)
submesh_D = SubMesh(mesh, subdomains, 2)

# --------------------------------------------------------------------
# Manufactured solution (matches your model & interface data)
# u_S(x,y) = [ w'(y) cos x , w(y) sin x ]
# p_D(x,y) = rho g exp(y) sin x
#
# with  w(y) = −K − (g y)/(2ν) + (K/2 − αg/(4ν²)) y²
# and  w'(y) = d/dy of the above.
# --------------------------------------------------------------------
# Build w and w' in Python (used inside Expressions)
A = -K
B = -g/(2.0*nu)
C = (K/2.0) - (alpha*g)/(4.0*nu**2)

# w(y) = A + B*y + C*y^2
# w'(y) = B + 2*C*y

# Vector Expression for u_S on Ω_S
uS_expr = Expression(
    (
        "(B + 2.0*C*x[1]) * cos(x[0])",     # u_x = w'(y) cos x
        "(A + B*x[1] + C*x[1]*x[1]) * sin(x[0])"  # u_y = w(y) sin x
    ),
    A=A, B=B, C=C, degree=4
)

# Scalar Expression for p_D on Ω_D
pD_expr = Expression("rho*g*exp(x[1])*sin(x[0])", rho=rho, g=g, degree=4)

# (Optional) body force b for reference — matches your formula
bx_expr = Expression(
    "((nu*K - (alpha*g)/(2.0*nu)) * x[1] - g/2.0) * cos(x[0])",
    nu=nu, K=K, alpha=alpha, g=g, degree=4
)
by_expr = Expression(
    "(((nu*K)/2.0 - (alpha*g)/(4.0*nu)) * x[1]*x[1] "
    "- (g/2.0) * x[1] + ((alpha*g)/(2.0*nu) - 2.0*nu*K)) * sin(x[0])",
    nu=nu, K=K, alpha=alpha, g=g, degree=4
)

# --------------------------------------------------------------------
# Interpolate on appropriate spaces and save to XDMF
# --------------------------------------------------------------------
# Stokes velocity on Ω_S
VS = VectorFunctionSpace(submesh_S, "CG", 2)
uS = interpolate(uS_expr, VS)

# Darcy pressure on Ω_D
QD = FunctionSpace(submesh_D, "CG", 2)
pD = interpolate(pD_expr, QD)

# Write to XDMF
with XDMFFile(submesh_S.mpi_comm(), "stokes_velocity.xdmf") as xdmf_u:
    xdmf_u.write(uS)

with XDMFFile(submesh_D.mpi_comm(), "darcy_pressure.xdmf") as xdmf_p:
    xdmf_p.write(pD)

# --------------------------------------------------------------------
# (Optional) quick sanity checks printed at a couple of points
# --------------------------------------------------------------------
if MPI.rank(MPI.comm_world) == 0:
    print("Saved: stokes_velocity.xdmf (on Ω_S) and darcy_pressure.xdmf (on Ω_D)")
    # Example values near interface y=0
    for xval in (0.5, 1.0, 2.0):
        try:
            print("u_S({:.2f}, 0.5) = {}".format(xval, uS(Point(xval, 0.5))))
        except:
            pass
        try:
            print("p_D({:.2f}, -0.5) = {:.6f}".format(xval, pD(Point(xval, -0.5))))
        except:
            pass