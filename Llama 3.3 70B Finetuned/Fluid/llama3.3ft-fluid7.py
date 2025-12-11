from dolfin import *
import matplotlib.pyplot as plt

# Geometry: channel with hole
L, H = 2.2, 0.41
R    = 0.05
domain = mshr.Rectangle(Point(0, 0), Point(L, H)) - mshr.Circle(Point(0.20, 0.20), R)
mesh   = mshr.generate_mesh(domain, 90)

# Function space: P2–P1
V = VectorFunctionSpace(mesh, "CG", 2); Q = FunctionSpace(mesh, "CG", 1)
ME = MixedElement([V.ufl_element(), Q.ufl_element()]); W = FunctionSpace(mesh, ME)

# No-slip on walls and hole
def walls(x, on_boundary):
    return (near(x[0], 0) or near(x[1], 0) or near(x[1], H)) and on_boundary

def hole_bndry(x, on_boundary):
    return on_boundary and sqrt((x[0] - 0.20)**2 + (x[1] - 0.20)**2) < R + DOLFIN_EPS

bcs = [
    DirichletBC(W.sub(0), Expression(("6*ubar*x[1]*(H-x[1])/H/H", "0.0"),
                                    ubar=0.2, H=H, degree=2),
                 "near(x[0], 0)"),
    DirichletBC(W.sub(0).sub(1), Constant(0.0), walls, method="pointwise"),
    DirichletBC(W.sub(0), Constant((0.0, 0.0)), hole_bndry, method="pointwise")
]

# Parameters
mu   = Constant(0.001)
rho  = Constant(1.0)
ubar = Constant(0.2)
D    = Constant(2*R)

# Variational formulation
(u, p) = TrialFunctions(W); (v, q) = TestFunctions(W)
f      = Constant((0.0, 0.0))
a = (mu*inner(grad(u), grad(v)) - div(v)*p + q*div(u))*dx
L = inner(f, v)*dx

# Solve
w = Function(W); solve(a == L, w, bcs)

u_soln, p_soln = w.split()
drag_vec = assemble(dot(sigma(u_soln, p_soln), n)*ds(3))
F_D      = sqrt(drag_vec[0]**2 + drag_vec[1]**2)
C_D      = 2*F_D/(rho*ubar**2*D)

print(f"Drag coefficient C_D = {C_D:.4f}")

# Visualization
u_mag = project(sqrt(dot(u_soln, u_soln)), FunctionSpace(mesh, "CG", 1))
plot_object = plot(u_mag, title="|u|")
plt.colorbar(plot_object)
plt.savefig("q7_speed.png")

file = XDMFFile("q7_soln.xdmf"); file.parameters["flush_output"] = True
file.write(u_soln, 0.0); file.write(p_soln, 0.0)