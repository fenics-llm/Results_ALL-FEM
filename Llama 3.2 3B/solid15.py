from fenics import *

# Create a mesh function for the domain
mesh = RectangleMesh(Point(0, 0), Point(1.2, 0.2), 100, 100)

# Define the boundary conditions
left_bc = DirichletBC(Mesh(), Constant(0), 'x=0')
right_bc = DirichletBC(Mesh(), Constant(0.012), 'x=1.2')
top_bc = DirichletBC(Mesh(), Constant(0), 'y=0')
bottom_bc = DirichletBC(Mesh(), Constant(0), 'y=0')

# Define the material parameters
lambda_ = 5.769e6
mu = 3.846e6

# Step 2: Implement the model and solve for u
def strain(u):
    F = gradient(u)
    E = 0.5 * (F.T * F - Identity(2))
    return E

def stress(E):
    lambda_ = 5.769e6
    mu = 3.846e6
    S = lambda_ * trace(E) * Identity(2) + 2 * mu * E
    return S

V = FunctionSpace(mesh, 'P', 2)
u = Function(V)

# Step 3: Assemble the equations and solve for u
def assemble_equations(u):
    F = gradient(u)
    E_val = strain(u)
    S_val = stress(E_val)
    sigma_val = (1 / det(F)) * F * S_val * F.T

    # Assemble the equations
    eqs = []
    for i in range(2):
        eqs.append(dot(grad(u), grad(v)) - 0.5 * dot(F.T, F) * v, u[i])
        eqs.append(dot(sigma_val, v) - f, v)
    return eqs

# Step 4: Implement the boundary conditions
def apply_bc(eqs, u):
    for i in range(2):
        eqs[i].sub(0).set_value(u_left_bc.evaluate(u))
    for i in range(2):
        eqs[2 + i].sub(0).set_value(u_right_bc.evaluate(u))

# Step 5: Implement the load stepping and Newton iterations
def solve_equations(eqs, u):
    # Solve the equations using Newton's method
    u_val = Function(V)
    for i in range(100):
        u_val.assign(u)
        E_val = strain(u_val)
        S_val = stress(E_val)
        sigma_val = (1 / det(F)) * F * S_val * F.T

        # Assemble the equations
        eqs_new = []
        for i in range(2):
            eqs_new.append(dot(grad(u_val), grad(v)) - 0.5 * dot(F.T, F) * v, u[i])
            eqs_new.append(dot(sigma_val, v) - f, v)

        # Apply the boundary conditions
        apply_bc(eqs_new, u_val)

        # Solve the new equations
        solve(eqs_new, u_val)
    return u_val

# Step 6: Implement the load stepping and stop when max principal Green–Lagrange strain E_max ≤ 0.03
def load_stepping(u):
    f = Constant(0)
    E_max = 0
    for i in range(100):
        # Solve the equations using Newton's method
        u_val = solve_equations(assemble_equations(u, E, S, sigma), u)

        # Compute the principal Green–Lagrange strains
        E_val = strain(u_val)
        E_max_val = max(E_val.eigenvalues())
        if E_max_val <= 0.03:
            return u_val
        else:
            f += Constant(1e-6) * (E_max_val - 0.03)

    # If the load stepping fails, return None
    return None

# Step 7: Save a plot of the deformed configuration as q15_def.png
def save_plot(u):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.cm import get_cmap

    # Create a color map for the maximum principal value E_max
    cmap = LinearSegmentedColormap.from_list("", ["blue", "red"])
    norm = MinMaxNorm()

    # Save a plot of the deformed configuration as q15_def.png
    u.plot()
    plt.savefig("q15_def.png")

# Step 8: Compute principal Green–Lagrange strains; save a color map of the maximum principal value E_max as q15_Emax.png
def compute_principal_strains(u):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.cm import get_cmap

    # Create a color map for the maximum principal value E_max
    cmap = LinearSegmentedColormap.from_list("", ["blue", "red"])
    norm = MinMaxNorm()

    # Compute the principal Green–Lagrange strains
    E_val = strain(u)
    E_max_val = max(E_val.eigenvalues())

    # Save a color map of the maximum principal value E_max as q15_Emax.png
    plt.imshow(E_val, cmap=cmap, norm=norm)
    plt.savefig("q15_Emax.png")

# Step 9: Define s = S − (1/3) tr(S) I and σ_vm(S) = sqrt(1.5 * (s:s)); save a color map as q15_vmS.png
def define_s_and_sigma_vm(u):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.cm import get_cmap

    # Define s = S − (1/3) tr(S) I and σ_vm(S) = sqrt(1.5 * (s:s))
    S_val = stress(strain(u))
    lambda_ = 5.769e6
    mu = 3.846e6
    sigma_vm = np.sqrt(1.5 * (S_val - 0.5 * trace(S_val) * Identity(2)) ** 2)

    # Save a color map as q15_vmS.png
    plt.imshow(sigma_vm, cmap=get_cmap("RdYlGn"))
    plt.savefig("q15_vmS.png")

# Step 10: Export the final displacement u and E_max in XDMF format (q15_u.xdmf, q15_Emax.xdmf)
def export_final_displacement(u):
    import xdmf

    # Create an XDMF file
    file = xdmf.XDMFFile("q15_u.xdmf")
    file.addVariable("u", u)

    # Export the final displacement u and E_max in XDMF format (q15_Emax.xdmf)
    file.addVariable("E_max", strain(u).eigenvalues())
    file.close()

# Step 11: Run the load stepping
load_stepping = load_stepping

# Step 12: Save a plot of the deformed configuration as q15_def.png
save_plot = save_plot

# Step 13: Compute principal Green–Lagrange strains; save a color map of the maximum principal value E_max as q15_Emax.png
compute_principal_strains = compute_principal_strains

# Step 14: Define s = S − (1/3) tr(S) I and σ_vm(S) = sqrt(1.5 * (s:s)); save a color map as q15_vmS.png
define_s_and_sigma_vm = define_s_and_sigma_vm

# Step 15: Export the final displacement u and E_max in XDMF format (q15_u.xdmf, q15_Emax.xdmf)
export_final_displacement = export_final_displacement

# Step 16: Run the load stepping
u_val = solve_equations(assemble_equations(u, E, S, sigma), u)

# Step 17: Save a plot of the deformed configuration as q15_def.png
save_plot(u_val)

# Step 18: Compute principal Green–Lagrange strains; save a color map of the maximum principal value E_max as q15_Emax.png
compute_principal_strains(u_val)

# Step 19: Define s = S − (1/3) tr(S) I and σ_vm(S) = sqrt(1.5 * (s:s)); save a color map as q15_vmS.png
define_s_and_sigma_vm(u_val)

# Step 20: Export the final displacement u and E_max in XDMF format (q15_u.xdmf, q15_Emax.xdmf)
export_final_displacement(u_val)

print("TERMINATE")