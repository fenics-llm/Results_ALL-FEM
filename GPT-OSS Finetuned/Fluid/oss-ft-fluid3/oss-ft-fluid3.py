# filename: q3_stokes.py
#!/usr/bin/env python3
import dolfin as dl
import matplotlib.pyplot as plt

# Mesh
mesh = dl.UnitSquareMesh(96, 96)

# Taylor–Hood mixed space (P2 velocity, P1 pressure)
V = dl.VectorFunctionSpace(mesh, "Lagrange", 2)   # velocity (P2)
Q = dl.FunctionSpace(mesh, "Lagrange", 1)         # pressure (P1)
W = dl.FunctionSpace(mesh, V.ufl_element() * Q.ufl_element())  # mixed

# Boundary markers
tol = 1E-14
def left(x, on):   return on and dl.near(x[0], 0.0, tol)
def right(x, on):  return on and dl.near(x[0], 1.0, tol)
def bottom(x, on): return on and dl.near(x[1], 0.0, tol)
def top(x, on):    return on and dl.near(x[1], 1.0, tol)

# Boundary conditions (apply to velocity subspace of W)
noslip = dl.Constant((0.0, 0.0))
lid    = dl.Constant((1.0, 0.0))

bcs = [
    dl.DirichletBC(W.sub(0), noslip, left),
    dl.DirichletBC(W.sub(0), noslip, right),
    dl.DirichletBC(W.sub(0), noslip, bottom),
    dl.DirichletBC(W.sub(0), lid,    top)
]

# Variational problem for steady Stokes flow
(u, p) = dl.TrialFunctions(W)
(v, q) = dl.TestFunctions(W)

mu = dl.Constant(1.0)   # dynamic viscosity

a = (2*mu*dl.inner(dl.sym(dl.grad(u)), dl.sym(dl.grad(v))) - dl.div(v)*p + q*dl.div(u)) * dl.dx
L = dl.Constant(0.0) * dl.inner(v, dl.Constant((0.0, 0.0))) * dl.dx

# Assemble system and apply Dirichlet BCs
A, b = dl.assemble_system(a, L, bcs)

# Solve linear system (use default solver; MUMPS may not be available)
null_vec = dl.Vector()
A.init_vector(null_vec, 1)
null_vec.zero()
for d in W.sub(1).dofmap().dofs():
    null_vec[d] = 1.0
nullspace = dl.VectorSpaceBasis([null_vec])
dl.as_backend_type(A).set_nullspace(nullspace)
nullspace.orthogonalize(b)

w = dl.Function(W)
dl.solve(A, w.vector(), b)

# Split solution into velocity and pressure
(u_sol, p_sol) = w.split()
u_sol.rename("velocity", "velocity")
p_sol.rename("pressure", "pressure")

# Enforce zero-mean pressure (remove nullspace)
dx = dl.dx(domain=mesh)
area = dl.assemble(dl.Constant(1.0) * dx)
p_mean = dl.assemble(p_sol * dx) / area
p_sol.vector().axpy(-p_mean, p_sol.vector())

# Write solution to XDMF
xdmf = dl.XDMFFile(mesh.mpi_comm(), "q3_soln.xdmf")
xdmf.write(u_sol)
xdmf.write(p_sol)
xdmf.close()

# Compute speed magnitude and save PNG plot
speed = dl.sqrt(dl.inner(u_sol, u_sol))
plt.figure()
dl.plot(speed, title="Speed |u|", cmap="viridis")
plt.savefig("q3_speed.png")