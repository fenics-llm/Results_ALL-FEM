from dolfin import *
import numpy as np, matplotlib.pyplot as plt

# Geometry
H = 0.2; Lf = 0.6; Lp = 1.0 - Lf
mesh = RectangleMesh(Point(0, 0), Point(Lp, H), 40, 20)

# Function spaces
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
ME = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, ME)

# Boundary markers
class Inlet(SubDomain):
    def inside(self, x, on_boundary): return near(x[0], 0.0) and on_boundary

class Outlet(SubDomain):
    def inside(self, x, on_boundary): return near(x[0], Lp) and on_boundary

class Walls(SubDomain):
    def inside(self, x, on_boundary): return (near(x[1], 0.0) or near(x[1], H)) and on_boundary

class Interface(SubDomain):
    def inside(self, x, on_boundary): return near(x[0], Lf) and on_boundary

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
Inlet().mark(boundaries, 1); Outlet().mark(boundaries, 2); Walls().mark(boundaries, 3)

# Boundary conditions
u_bar = 0.1; mu = Constant(0.02); K = Constant(1e-6)
ux_expr = Expression("6*U*x[1]*(H-x[1])/pow(H,2)", U=u_bar, H=H, degree=2)
inlet_bc = DirichletBC(W.sub(0), as_vector([ux_expr, 0.0]), boundaries, 1)

# Variational formulation
(u, p) = TrialFunctions(W); (v, q) = TestFunctions(W)
f = Constant(0.0)
a_f = (mu*inner(grad(u), grad(v)) - div(v)*p + q*div(u))*dx
L_f = f*v*dx

# Darcy in porous medium
u_p, p_p = u, p
a_p = (K*inner(grad(p_p), grad(q)) + div(u_p)*q)*dx
L_p = Constant(0.0)*q*dx

# Interface conditions
n = FacetNormal(mesh)
h = CellDiameter(mesh); beta = 10.0/h('+')
sigma_n = dot(sigma(u, p), n('+'))
a_int = (beta*dot(u('+'), n('+'))*dot(v('+'), n('+')) +
         beta*dot(u('-'), n('-'))*dot(v('-'), n('-')))*ds(4)
L_int = -beta*p('+')*dot(v('+'), n('+'))*ds(4) \
        -beta*p('-')*dot(v('-'),n('-))*ds(4)

# Outlet pressure
p_out = Constant(0.0); outlet_bc = DirichletBC(W.sub(1), p_out, boundaries, 2)

# Solve the coupled problem
a = a_f + a_p + a_int
L = L_f + L_p + L_int

w = Function(W)
solve(a == L, w, [inlet_bc, outlet_bc])

u_sol, p_sol = w.split()

# Postprocessing: extract interface profiles
interface_pts = np.linspace(0.0, H, 100); ux_values = []; uy_values = []
for y in interface_pts:
    pt = Point(Lf, y)
    ux_values.append(u_sol(pt)[0])
    uy_values.append(u_sol(pt)[1])

np.savetxt("q15_interface.csv", np.column_stack((interface_pts, ux_values, uy_values)),
           header="y  u_x  u_y", comments="")

# Visualization
plt.figure(figsize=(8,3))
p = plot(p_sol, title="Pressure p"); plt.colorbar(p)
plt.savefig("q15_p.png", dpi=150)

File("q15_solution.xdmf") << w