# filename: elastoplastic_quarter_plate.py
from dolfin import *
import mshr
import numpy as np
from ufl_legacy import Identity, tr, as_tensor   # legacy UFL API for current FEniCS

# -------------------------------------------------------------------
# Geometry (mm → m conversion later)
# -------------------------------------------------------------------
R_hole = 50.0          # mm
Lx = 100.0             # mm
Ly = 180.0             # mm

# Create quarter plate with a quarter circular hole at the origin
domain = mshr.Rectangle(Point(0.0, 0.0), Point(Lx, Ly)) \
         - mshr.Circle(Point(0.0, 0.0), R_hole, 64)
mesh = mshr.generate_mesh(domain, 64)   # increase resolution if needed

# -------------------------------------------------------------------
# Unit conversion (mm → m)
# -------------------------------------------------------------------
mm = 1e-3
R_hole *= mm
Lx     *= mm
Ly     *= mm

# -------------------------------------------------------------------
# Function spaces
# -------------------------------------------------------------------
V  = VectorFunctionSpace(mesh, "CG", 2)          # displacement
V0 = TensorFunctionSpace(mesh, "DG", 0)          # cell‑wise tensors (strain, stress, plastic strain)

# Primary unknown
u = Function(V, name="Displacement")

# Internal variables (piecewise constant per cell)
eps_p = Function(V0, name="PlasticDeviatoricStrain")   # deviatoric plastic strain
eqp   = Function(FunctionSpace(mesh, "DG", 0), name="EqPlasticStrain")  # equivalent plastic strain

# -------------------------------------------------------------------
# Material parameters (SI)
# -------------------------------------------------------------------
lam = 19.44e9          # λ  [Pa]
mu  = 29.17e9          # μ  [Pa]
sigma_y = 243e6        # yield stress [Pa]

# -------------------------------------------------------------------
# Helper functions (UFL)
# -------------------------------------------------------------------
def eps(u):
    """Small strain tensor ε = sym(∇u)"""
    return sym(grad(u))

def deviatoric(t):
    """Deviatoric part of a second‑order tensor"""
    return t - (1.0/3.0)*tr(t)*Identity(2)

# -------------------------------------------------------------------
# Return‑mapping (NumPy, operates on cell‑wise strain)
# -------------------------------------------------------------------
def stress_update(eps_el_arr):
    """
    eps_el_arr : (ncell,2,2) elastic strain (total - plastic)
    Returns
        sigma_arr   : (ncell,2,2) Cauchy stress
        eps_p_inc   : (ncell,2,2) increment of plastic deviatoric strain
    """
    ncell = eps_el_arr.shape[0]

    # ----- Elastic trial stress ------------------------------------
    tr_eps = np.trace(eps_el_arr, axis1=1, axis2=2)          # (ncell,)
    sigma_tr = np.empty_like(eps_el_arr)
    for a in range(2):
        for b in range(2):
            sigma_tr[:, a, b] = lam * tr_eps * (1.0 if a == b else 0.0) \
                                + 2.0 * mu * (eps_el_arr[:, a, b] -
                                              (1.0/3.0) * tr_eps * (1.0 if a == b else 0.0))

    # ----- Deviatoric trial stress --------------------------------
    s_tr = sigma_tr - (1.0/3.0) * np.trace(sigma_tr, axis1=1, axis2=2)[:, None, None] * np.eye(2)
    q_tr = np.sqrt(1.5 * np.einsum('...ij,...ij', s_tr, s_tr))

    # Initialise outputs
    sigma = np.copy(sigma_tr)
    eps_p_inc = np.zeros_like(eps_el_arr)

    # ----- Yield check --------------------------------------------
    yielded = q_tr > sigma_y + 1e-12

    if np.any(yielded):
        # Plastic multiplier Δγ = (q_tr - σ_y) / (3 μ)
        dgamma = (q_tr[yielded] - sigma_y) / (3.0 * mu)

        # Unit normal to yield surface n = (3/2) s_tr / q_tr
        n = 1.5 * s_tr[yielded] / q_tr[yielded][:, None, None]

        # Increment of plastic deviatoric strain
        eps_p_inc[yielded] = dgamma[:, None, None] * n

        # Updated stress: σ = σ_tr - 2 μ Δγ n
        sigma[yielded] = sigma_tr[yielded] - 2.0 * mu * dgamma[:, None, None] * n

    return sigma, eps_p_inc

# -------------------------------------------------------------------
# Residual and Jacobian assembly (returns UFL forms)
# -------------------------------------------------------------------
def residual_and_jacobian(u_func):
    """
    Build the residual R(v) = ∫ σ : ε(v) dx
    and a Jacobian using the elastic tangent (sufficient for perfect plasticity).
    """
    v  = TestFunction(V)
    du = TrialFunction(V)

    # Total strain (UFL)
    eps_tot = eps(u_func)

    # Project strain to cell‑wise constant space to obtain a NumPy array
    eps_tot_proj = project(eps_tot, V0)
    eps_tot_arr = eps_tot_proj.vector().get_local().reshape((-1, 2, 2))

    # Plastic strain from previous iteration (NumPy)
    eps_p_arr = eps_p.vector().get_local().reshape((-1, 2, 2))

    # Elastic strain = total - plastic
    eps_el_arr = eps_tot_arr - eps_p_arr

    # Material update
    sigma_arr, eps_p_inc_arr = stress_update(eps_el_arr)

    # Store updated internal variables
    eps_p.vector()[:] = (eps_p_arr + eps_p_inc_arr).reshape(-1)

    # Equivalent plastic strain increment (for post‑processing only)
    d_eqp = np.sqrt(2.0/3.0) * np.linalg.norm(eps_p_inc_arr.reshape(-1, 4), axis=1)
    eqp.vector()[:] = eqp.vector().get_local() + d_eqp

    # Create a DG0 tensor Function for stress to use in the variational form
    sigma_func = Function(V0, name="Stress")
    sigma_func.vector()[:] = sigma_arr.reshape(-1)

    # Residual
    R = inner(sigma_func, eps(v)) * dx

    # Jacobian – elastic tangent (perfect plasticity)
    C_el = lam * tr(eps(du)) * Identity(2) + 2.0 * mu * deviatoric(eps(du))
    J = inner(C_el, eps(v)) * dx

    return R, J

# -------------------------------------------------------------------
# Boundary conditions
# -------------------------------------------------------------------
# Symmetry on x = 0  → u_x = 0
class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and on_boundary
left_bc = DirichletBC(V.sub(0), Constant(0.0), LeftBoundary())

# Symmetry on y = 0  → u_y = 0
class BottomBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0) and on_boundary
bottom_bc = DirichletBC(V.sub(1), Constant(0.0), BottomBoundary())

# Prescribed displacement on top edge y = 180 mm → (0, 1 mm)
class TopBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], Ly) and on_boundary
top_bc = DirichletBC(V, Constant((0.0, 1.0 * mm)), TopBoundary())

bcs = [left_bc, bottom_bc, top_bc]

# -------------------------------------------------------------------
# Non‑linear solver (single load step)
# -------------------------------------------------------------------
# Initial guess
u.assign(Constant((0.0, 0.0)))

tol = 1e-8
max_iter = 25

for it in range(max_iter):
    # Assemble residual and Jacobian with current internal variables
    R_form, J_form = residual_and_jacobian(u)

    A = assemble(J_form)
    b = assemble(-R_form)
    for bc in bcs:
        bc.apply(A, b)

    solve(A, u.vector(), b, "mumps")

    res_norm = b.norm('linf')
    print(f"Newton iteration {it+1}: residual = {res_norm:.3e}")

    if res_norm < tol:
        print("Converged.")
        break
else:
    print("Warning: Newton did not converge within the maximum number of iterations.")

# -------------------------------------------------------------------
# Save displacement to XDMF
# -------------------------------------------------------------------
xdmf = XDMFFile(mesh.mpi_comm(), "quarter_plate_displacement.xdmf")
xdmf.write(u)
xdmf.close()
print("Displacement saved to 'quarter_plate_displacement.xdmf'.")