from dolfin import *
from mshr   import Rectangle, generate_mesh

# 1. Geometry: strip (0,1)×(0,0.2)
mesh = generate_mesh(Rectangle(Point(0,0), Point(1,0.2)), 100)

# 2. Material: E=5e6, ν=0.49
E, nu = 5e6, 0.49

# 3. Boundary conditions:
bdr = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
bdr.set_all(0); eps = 1e-10
AutoSubDomain(lambda x,on: near(x[0],0,eps)).mark(bdr,1)   # left edge
AutoSubDomain(lambda x,on: near(x[0],1,eps)).mark(bdr,2)   # right edge
AutoSubDomain(lambda x,on: near(x[1],0,eps)).mark(bdr,3)   # bottom edge
AutoSubDomain(lambda x,on: near(x[1],0.2,eps)).mark(bdr,4)  # top edge

bc_left = DirichletBC(V, Constant((0,0)), bdr, 1)
bc_right = DirichletBC(V, Constant((0.03,0)), bdr, 2)

# 4. Variational problem: u_x=0 on left, u_y=0 on right
u = TrialFunction(V); v = TestFunction(V)
a = E*inner(grad(u),grad(v))*dx + nu*(div(v)*div(u)-tr(div(u))*tr(div(v)))*dx

# 5. Traction-free top/bottom
L = Constant((0,0))*ds(3) + Constant((0,0))*ds(4)

u_sol = Function(V); solve(a == L, u_sol, [bc_left,bc_right])

# 6. Output: displacement magnitude |u| as q10_disp.png
import matplotlib.pyplot as plt
plt.figure(); c=plot(u_sol,title="q10 - displacement magnitude"); plt.colorbar(c)
plt.savefig("q10_disp.png")

# 7. Save XDMF file for post-processing
File("q10_xdmf.xml") << u_sol