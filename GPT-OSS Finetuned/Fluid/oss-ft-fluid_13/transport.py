# filename: transport.py
from dolfin import *
import numpy as np

# -------------------  Parameters  -------------------
L = 1.0          # length (m)
H = 0.10         # height (m)
U_max = 0.75     # max velocity (m/s)
D = 1.0e-5       # diffusivity (m^2/s)

# -------------------  Mesh  -------------------
nx, ny = 100, 10
mesh = RectangleMesh(Point(0.0, 0.0), Point(L, H), nx, ny, "crossed")

# -------------------  Function space  -------------------
V = FunctionSpace(mesh, "CG", 1)

# -------------------  Velocity field  -------------------
u_expr = Expression(("U_max*4*x[1]*(H - x[1])/pow(H,2)", "0.0"),
                    U_max=U_max, H=H, degree=2)
V_vec = VectorFunctionSpace(mesh, "CG", 2)
u = interpolate(u_expr, V_vec)

# -------------------  Boundary conditions  -------------------
inlet  = CompiledSubDomain("near(x[0], 0.0) && on_boundary")
outlet = CompiledSubDomain("near(x[0], L)   && on_boundary", L=L)
walls  = CompiledSubDomain("near(x[1], 0.0) || near(x[1], H) && on_boundary", H=H)

bc_in  = DirichletBC(V, Constant(0.0), inlet)
bc_out = DirichletBC(V, Constant(1.0), outlet)
bcs = [bc_in, bc_out]   # walls have natural zero diffusive flux

# -------------------  Stabilization (SUPG)  -------------------
h = CellDiameter(mesh)
u_norm = sqrt(dot(u, u) + DOLFIN_EPS)
tau = h/(2.0*u_norm)   # streamline diffusion parameter (Function)

# -------------------  Variational problem  -------------------
c = TrialFunction(V)
v = TestFunction(V)

a = D*dot(grad(c), grad(v))*dx + dot(u, grad(c))*v*dx
Lrhs = Constant(0.0)*v*dx

# SUPG terms
a += tau*dot(u, grad(v))*dot(u, grad(c))*dx
Lrhs += tau*dot(u, grad(v))*Constant(0.0)*dx   # RHS zero (no source)

# -------------------  Solve  -------------------
c_sol = Function(V)
solve(a == Lrhs, c_sol, bcs,
      solver_parameters={"linear_solver": "bicgstab",
                         "preconditioner": "ilu"})

# -------------------  Output  -------------------
with XDMFFile(mesh.mpi_comm(), "concentration.xdmf") as xdmf:
    xdmf.write(c_sol, 0.0)