# filename: elastoplastic_plate.py
from dolfin import *
import mshr
import numpy as np

# --------------------------------------------------------------
# Geometry and mesh (quarter plate with quarter hole)
# --------------------------------------------------------------
Lx, Ly = 100.0, 180.0          # mm
R = 50.0                        # mm
domain = mshr.Rectangle(Point(0.0, 0.0), Point(Lx, Ly)) \
         - mshr.Circle(Point(0.0, 0.0), R, 64)
mesh = mshr.generate_mesh(domain, 64)

# --------------------------------------------------------------
# Material parameters (plane strain)
# --------------------------------------------------------------
E  = 200.0e3                    # MPa
nu = 0.3
lam = 1.944e4                   # λ (MPa)
mu  = 2.917e4                   # μ (MPa)
sigma_y = 243.0                 # MPa (yield stress)

# --------------------------------------------------------------
# Function spaces
# --------------------------------------------------------------
V = VectorFunctionSpace(mesh, "Lagrange", 2)   # displacement
W = FunctionSpace(mesh, "DG", 0)                # equivalent plastic strain εp
u = Function(V, name="Displacement")
v = TestFunction(V)

epsp = Function(W, name="PlasticStrain")       # current εp
epsp_old = Function(W, name="PlasticStrain_old")
epsp_old.vector()[:] = 0.0

# --------------------------------------------------------------
# Kinematics
# --------------------------------------------------------------
def eps(u):
    return sym(grad(u))

def sigma(u, epsp_val):
    e = eps(u)
    tr_e = tr(e)
    e_dev = e - (1./3)*tr_e*Identity(2)
    s_tr = 2*mu*e_dev
    q_tr = sqrt(3./2.*inner(s_tr, s_tr))
    fac = conditional(q_tr <= sigma_y, 1.0, sigma_y/q_tr)
    s = fac*s_tr
    return lam*tr_e*Identity(2) + s

# --------------------------------------------------------------
# Boundary conditions
# --------------------------------------------------------------
tol = 1E-6
def left(x, on):  return on and near(x[0], 0.0, tol)
def bottom(x, on):return on and near(x[1], 0.0, tol)
def top(x, on):   return on and near(x[1], Ly, tol)

bcs = [DirichletBC(V.sub(0), Constant(0.0), left),
       DirichletBC(V.sub(1), Constant(0.0), bottom)]

class TopDisp(UserExpression):
    def __init__(self, factor=0.0, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor
    def eval(self, values, x):
        values[0] = 0.0
        values[1] = self.factor
    def value_shape(self):
        return (2,)

u_top = TopDisp(degree=1)
bcs.append(DirichletBC(V, u_top, top))

# --------------------------------------------------------------
# Load stepping
# --------------------------------------------------------------
nsteps = 20
load_factor = 1.0/nsteps

dx = Measure('dx', domain=mesh)

# --------------------------------------------------------------
# Time stepping loop
# --------------------------------------------------------------
for step in range(1, nsteps+1):
    u_top.factor = step*load_factor

    F = inner(sigma(u, epsp_old), eps(v))*dx
    solve(F == 0, u, bcs)

    s_tr = 2*mu*(eps(u) - (1./3)*tr(eps(u))*Identity(2))
    q_trial = sqrt(3./2.*inner(s_tr, s_tr))
    delta_epsp_expr = conditional(q_trial > sigma_y,
                                   (q_trial - sigma_y)/(3*mu),
                                   0.0)
    delta_epsp = project(delta_epsp_expr, W)

    epsp.vector()[:] = epsp_old.vector() + delta_epsp.vector()
    epsp_old.assign(epsp)

    xdmf = XDMFFile(mesh.mpi_comm(),
                     "plate_disp_step{:02d}.xdmf".format(step))
    xdmf.write(u, step)
    xdmf.close()

    max_u = np.max(np.abs(u.vector().get_local()))
    print("Step {:2d}/{:2d} completed, max |u| = {:.4e} mm".format(
          step, nsteps, max_u))

print("All steps finished.")