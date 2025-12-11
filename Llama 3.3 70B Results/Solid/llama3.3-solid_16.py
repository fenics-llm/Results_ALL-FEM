from fenics import *

# Create mesh
mesh = RectangleMesh(Point(0, 0), Point(100, 180), 100, 180)

# Define function space
V = VectorFunctionSpace(mesh, 'P', 1)

# Define boundary conditions
u_D = Constant((0, 1))

def boundary(x, on_boundary):
    return on_boundary and x[1] == 180

bc = DirichletBC(V, u_D, boundary)

# Define symmetry boundary conditions
class SymmetryX(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0)

class SymmetryY(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0)

symmetry_x = SymmetryX()
symmetry_y = SymmetryY()

# Define material parameters
lmbda = 19.44e3  # GPa
mu = 29.17e3  # GPa
sigma_Y = 243e6  # Pa

# Define stress and strain functions
def sigma(u):
    return lmbda * div(u) * Identity(2) + 2 * mu * sym(grad(u))

def epsilon(u):
    return sym(grad(u))

# Compute deviatoric stress and strain
def deviatoric_stress(sigma):
    return sigma - (1/3) * tr(sigma) * Identity(2)

def deviatoric_strain(epsilon):
    return epsilon - (1/3) * tr(epsilon) * Identity(2)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)

a = inner(grad(u)[0, 0]*grad(v)[0, 0] + grad(u)[0, 1]*grad(v)[0, 1] + 
           grad(u)[1, 0]*grad(v)[1, 0] + grad(u)[1, 1]*grad(v)[1, 1], 1) + \
    lmbda * div(u) * div(v)
L = inner(Constant((0, 0)), v)

# Solve variational problem
u = Function(V)
A = assemble(a)
b = assemble(L)
bc.apply(A)
solve(A, u.vector(), b)

# Save solution to file
file = XDMXFile("displacement.xdmx")
file.write(u, 0)
