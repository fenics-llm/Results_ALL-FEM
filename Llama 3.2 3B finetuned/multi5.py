from dolfin import *
import mshr

mesh = mshr.generate_mesh(mshr.Rectangle(Point(0, 0), Point(6, 1)), 64)
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)

class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0)

class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 6.0)

inlet = Inlet()
outlet = Outlet()

boundaries.set_all(0)
inlet.mark(boundaries, 1)
outlet.mark(boundaries, 2)

ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

u, p = TrialFunctions(V), TestFunctions(Q)
v, q = Tensors(u), Tensors(q)

f = Constant((0.0, -rho_f*g[1]))

# Fluid
a_fluid = inner(mu_f*grad(u), grad(v))*dx + div(v)*p*dx + q*div(u)*dx

L_fluid = dot(f, v)*dx

# Solid
a_solid = E_s*(inner(grad(u), grad(v)) - nu_s*inner(grad(u), grad(w))*dx
              + mu_s*inner(grad(q), w)*dx)

L_solid = Constant(0.0)*w*dx

# Fluid-solid coupling
a_fluid += a_solid
L_fluid += L_solid

u_sol, p_sol = Function(V), Function(Q)
solve(a_fluid == L_fluid, u_sol, bcs=[dirichlet_bc(u_sol, "inlet")], solver_parameters={"linear_solver": "mumps"})
p_sol.vector()[:] = project(-div(u_sol), Q)