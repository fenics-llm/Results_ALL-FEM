# filename: quarter_hole_j2_plasticity_plane_strain.py
from dolfin import *
from mshr import Rectangle, Circle, generate_mesh

# -------------------------
# geometry (mm)
# -------------------------
Lx, Ly = 100.0, 180.0
R      = 50.0
res    = 80
domain = Rectangle(Point(0.0, 0.0), Point(Lx, Ly)) - Circle(Point(0.0, 0.0), R, 96)
mesh   = generate_mesh(domain, res)

# -------------------------
# function space
# -------------------------
V = VectorFunctionSpace(mesh, "CG", 2)

# -------------------------
# material (MPa) - plane strain
# -------------------------
lam = Constant(19440.0)   # 19.44 GPa
mu  = Constant(29170.0)   # 29.17 GPa
sigY = Constant(243.0)    # 243 MPa
I = Identity(2)

# bulk modulus K = lam + 2/3 mu (appears in sigma = K*tr(eps)*I + s)
K = lam + (2.0/3.0)*mu

# -------------------------
# kinematics
# -------------------------
def eps(u):
    return sym(grad(u))

def dev(A):
    return A - (tr(A)/3.0)*I

def j2_return_map(eps_u):
    e_dev  = dev(eps_u)
    s_tr   = 2.0*mu*e_dev
    q_tr   = sqrt(1.5)*sqrt(inner(s_tr, s_tr) + DOLFIN_EPS)
    r      = conditional(le(q_tr, sigY), Constant(1.0), sigY/q_tr)
    s_alg  = r*s_tr
    # volumetric + deviatoric stress
    sigma  = K*tr(eps_u)*I + s_alg
    return sigma

# -------------------------
# boundary markers
# -------------------------
class XSym(SubDomain):
    def inside(self, x, on_bnd):
        return on_bnd and near(x[0], 0.0)

class YSym(SubDomain):
    def inside(self, x, on_bnd):
        return on_bnd and near(x[1], 0.0)

class Top(SubDomain):
    def inside(self, x, on_bnd):
        return on_bnd and near(x[1], Ly)

class Right(SubDomain):
    def inside(self, x, on_bnd):
        return on_bnd and near(x[0], Lx)

class Hole(SubDomain):
    def inside(self, x, on_bnd):
        # quarter arc: x^2 + y^2 = R^2 within rectangle boundaries
        return on_bnd and near(x[0]*x[0] + x[1]*x[1], R*R, 1e-6)

x_sym = XSym()
y_sym = YSym()
top   = Top()
right = Right()
hole  = Hole()

# -------------------------
# trial/test and unknowns
# -------------------------
u  = Function(V)
du = TrialFunction(V)
v  = TestFunction(V)

# -------------------------
# loading
# -------------------------
u_top_total = 1.0  # mm
nsteps = 20
load_incs = [u_top_total*(i+1)/float(nsteps) for i in range(nsteps)]

# Dirichlet BCs
bc_xsym = DirichletBC(V.sub(0), Constant(0.0), x_sym)     # u_x = 0 on x=0
bc_ysym = DirichletBC(V.sub(1), Constant(0.0), y_sym)     # u_y = 0 on y=0
u_top = Constant(0.0)
bc_top_u = DirichletBC(V, as_vector([Constant(0.0), u_top]), top)  # u=(0, Uy) on y=Ly
bcs = [bc_xsym, bc_ysym, bc_top_u]

# -------------------------
# weak form (quasi-static, traction-free elsewhere)
# -------------------------
sigma = j2_return_map(eps(u))
res = inner(sigma, eps(v))*dx
Jac = derivative(res, u, du)

problem = NonlinearVariationalProblem(res, u, bcs, Jac)
solver = NonlinearVariationalSolver(problem)
prm = solver.parameters
prm["nonlinear_solver"] = "newton"
prm["newton_solver"]["maximum_iterations"] = 20
prm["newton_solver"]["relative_tolerance"] = 1e-8
prm["newton_solver"]["absolute_tolerance"] = 1e-9
prm["newton_solver"]["linear_solver"] = "mumps"

# -------------------------
# output
# -------------------------
xdmf = XDMFFile(mesh.mpi_comm(), "quarter_hole_j2_disp.xdmf")
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True

for Uy in load_incs:
    u_top.assign(Uy)
    solver.solve()
    xdmf.write(u, Uy)

xdmf.close()