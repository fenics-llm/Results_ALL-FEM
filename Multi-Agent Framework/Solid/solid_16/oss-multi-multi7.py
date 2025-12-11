from dolfin import *
import mshr
import numpy as np

# ---------------------------------------------------------------
# 1. Geometry & mesh
# ---------------------------------------------------------------
Lx, Ly, R = 0.100, 0.180, 0.050          # metres
resolution = 80                             # mesh density (increase for finer results)

domain = mshr.Rectangle(Point(0.0, 0.0), Point(Lx, Ly)) \
         - mshr.Circle(Point(0.0, 0.0), R, resolution)
mesh = mshr.generate_mesh(domain, resolution)

# ---------------------------------------------------------------
# 2. Function spaces
# ---------------------------------------------------------------
Ve = VectorElement("Lagrange", mesh.ufl_cell(), 2)   # P2 for displacement (plane strain)
V  = FunctionSpace(mesh, Ve)

# Plastic strain (deviatoric) stored at cell centres (DG0 tensor)
T0 = TensorElement("DG", mesh.ufl_cell(), 0)
Vpl = FunctionSpace(mesh, T0)

# ---------------------------------------------------------------
# 3. Material parameters
# ---------------------------------------------------------------
lam = 1.944e10
mu  = 2.917e10
sigma_y = 2.43e8

# ---------------------------------------------------------------
# 4. Boundary definitions
# ---------------------------------------------------------------
tol = 1E-10

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], Ly, tol)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], Lx, tol)

class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, tol)

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0, tol)

top    = Top()
right  = Right()
left   = Left()
bottom = Bottom()

# Dirichlet BCs
zero = Constant((0.0, 0.0))
bcs = [DirichletBC(V.sub(0), Constant(0.0), left),
       DirichletBC(V.sub(1), Constant(0.0), bottom)]

# Prescribed displacement on the top edge (incremental)
u_top = Expression(("0.0", "t"), t=0.0, degree=1)
bcs.append(DirichletBC(V, u_top, top))

# ---------------------------------------------------------------
# 5. Plastic strain storage (initially zero)
# ---------------------------------------------------------------
eps_p     = Function(Vpl)          # current plastic deviatoric strain (cellwise)
eps_p_old = Function(Vpl)          # plastic strain from previous load step

# ---------------------------------------------------------------
# 6. Kinematics & trial stress
# ---------------------------------------------------------------
def eps(v):
    return sym(grad(v))

# ---------------------------------------------------------------
# 7. Elastoplastic update (cellwise) – corrected version
# ---------------------------------------------------------------
def plastic_update(u):
    """Return-mapping for perfect von Mises plasticity (plane strain).

    The plastic deviatoric strain εᵖ is stored cell-wise (DG0 tensor).
    After looping over all cells we copy the updated values back
    to the Function `eps_p` via a full-slice assignment.
    """
    # total strain (cellwise) – project onto the DG0 tensor space
    eps_tot = project(eps(u), Vpl)          # ε = sym(∇u)

    # extract underlying NumPy arrays
    eps_tot_vec   = eps_tot.vector().get_local()
    eps_p_old_vec = eps_p_old.vector().get_local()
    ncells = mesh.num_cells()

    # array that will hold the new plastic strain (flattened 2×2 per cell)
    eps_p_new_vec = np.copy(eps_p_old_vec)

    for cell in cells(mesh):
        cid = cell.index()

        # reconstruct strain tensor at cell centre from flattened vector
        e = np.array([[eps_tot_vec[4*cid+0],   0.5*(eps_tot_vec[4*cid+1]+eps_tot_vec[4*cid+2])],
                      [0.5*(eps_tot_vec[4*cid+1]+eps_tot_vec[4*cid+2]), eps_tot_vec[4*cid+3]]])

        # volumetric part and deviatoric strain
        eps_vol = (1.0/3.0)*np.trace(e)*np.eye(2)
        e_dev   = e - eps_vol

        # previous plastic deviatoric strain for this cell
        ep_old = np.array([[eps_p_old_vec[4*cid+0], 0.5*(eps_p_old_vec[4*cid+1]+eps_p_old_vec[4*cid+2])],
                           [0.5*(eps_p_old_vec[4*cid+1]+eps_p_old_vec[4*cid+2]), eps_p_old_vec[4*cid+3]]])

        # trial deviatoric stress
        s_tr = 2.0*mu*(e_dev - ep_old)

        # von Mises equivalent stress
        q_tr = np.sqrt(1.5*np.tensordot(s_tr, s_tr))

        if q_tr <= sigma_y + 1e-12:               # elastic step → keep old εᵖ
            ep_new = ep_old
        else:                                      # plastic correction
            delta_gamma = (q_tr - sigma_y) / (3.0*mu)
            n = s_tr / q_tr                       # flow direction
            ep_new = ep_old + delta_gamma * n

        # store updated plastic strain (flattened) back into global array
        eps_p_new_vec[4*cid+0] = ep_new[0,0]
        eps_p_new_vec[4*cid+1] = ep_new[0,1]
        eps_p_new_vec[4*cid+2] = ep_new[1,0]
        eps_p_new_vec[4*cid+3] = ep_new[1,1]

    # copy the full array back to the Function (full-slice is allowed)
    eps_p.vector().set_local(eps_p_new_vec)
    eps_p.vector().apply("insert")

# ---------------------------------------------------------------
# 8. Residual and Jacobian (using current plastic strain)
# ---------------------------------------------------------------
u = Function(V)          # current displacement
du = TrialFunction(V)
v  = TestFunction(V)

def sigma(u):
    e = eps(u)
    eps_vol = (1.0/3.0)*tr(e)*Identity(2)
    e_dev   = e - eps_vol
    s = 2.0*mu*(e_dev - eps_p)          # stress with current plastic strain
    return (lam + 2.0*mu/3.0)*tr(e)*Identity(2) + s

R = inner(sigma(u), eps(v))*dx
J = derivative(R, u, du)

# ---------------------------------------------------------------
# 9. Incremental loading
# ---------------------------------------------------------------
n_steps = 20
u_top.t = 0.0
load_increment = 1.0e-3 / n_steps   # 1 mm total

for step in range(1, n_steps+1):
    u_top.t = step*load_increment
    eps_p_old.assign(eps_p)          # store plastic strain from previous step

    # Newton iterations
    tol_newton = 1e-6
    max_iter = 25
    for it in range(max_iter):
        plastic_update(u)             # update stress based on current u
        A = assemble(J)
        b = assemble(R)
        [bc.apply(A, b) for bc in bcs]
        solve(A, u.vector(), b, "lu")
        residual = b.norm("l2")
        if residual < tol_newton:
            break
    print("Step {:2d} / {:2d}  |  u_top = {:.6f} m  |  Newton it = {:2d}  |  residual = {:.2e}"
          .format(step, n_steps, u_top.t, it+1, residual))

# ---------------------------------------------------------------
# 10. Save results
# ---------------------------------------------------------------
xdmf = XDMFFile(mesh.mpi_comm(), "quarter_plate_displacement.xdmf")
xdmf.write(u)