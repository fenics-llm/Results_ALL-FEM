from dolfin import *
import mshr

# Geometry
H = 1.0
mesh = mshr.Rectangle(Point(-3*H,0), Point(20*H,H))
mesh2 = mshr.Rectangle(Point(0,0), Point(0,2*H))

domain = mesh + mesh2 - (mesh3 - Point(0,0)) # union of upstream and downstream channels
# outlet is the intersection of downstream channel and x=20H plane

# Mark boundaries for no-slip conditions
boundaries = MeshFunction("size_t", domain, 1, 0)
boundaries.set_all(2) # default value: 2 = interior
AutoSubDomain(lambda x,on: near(x[0],-3*H,1e-10)).mark(boundaries,1)   # bottom wall
AutoSubDomain(lambda x,on: near(x[0],0,1e-10)).mark(boundaries,1)       # step wall (x=0)
AutoSubDomain(lambda x,on: near(x[0],20*H,1e-10)).mark(boundaries,2)    # outlet
AutoSubDomain(lambda x,on: near(x[1],0,1e-10)).mark(boundaries,3)       # top wall

ds = Measure("ds", domain=domain, subdomain_data=boundaries)

# Function space
V = VectorFunctionSpace(domain,"P",2)
u, p = TrialFunctions(V), TestFunctions(V)

# Inlet velocity
U_bar = 1.0
def inlet_velocity(x):
    y = x[1]
    return as_vector([6*U_bar*y/H*(1-y/H), 0])

# Variational form
f = Constant((0.0, -9.81))
a = (u[0]*w[0] + u[1]*w[1]) * dx \
   + mu*inner(grad(u), grad(w)) * dx \
   - p*w[0]*(ds(2) + ds(3)) \
   - p*w[1]*(ds(1) + ds(3))
L = f[0]*w[0] + f[1]*w[1] * dx

# Inlet velocity
u_in = interpolate(inlet_velocity, V)

# Solve
uh = Function(V)
solve(a == L, uh, [u_in, Constant(0.0)])

# Save
plt.figure()
p = project(uh[1], FunctionSpace(domain,"P",1))
plot(p); plt.colorbar(); plt.title("Pressure (Pa)"); plt.savefig("q6_p.png")
plt.close()

# Compute wall shear stress at top wall
tau_w = project(-uh[0]*uh[1]*(ds(3)), FunctionSpace(domain,"P",1))

# Find re-attachment point
print("Re-attachment point: x = %.2f m" % tau_w.vector().argmin())