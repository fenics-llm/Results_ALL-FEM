
# filename: cylinder_vms.py

from dolfin import *
import mshr
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Mesh and function space
    mesh = mshr.RectangleMesh(mshr.Rectangle(Point(-30, -30), Point(30, 30)), 64, 64)
    V = VectorFunctionSpace(mesh, "P", 1)

    # Mark cylinder
    def inside_cylinder(x, on):
        return on and (
            (x[0] - 0.5 * 1)**2 + (x[1] - 0) ** 2 <= 0.25**2
        )

    class Cylinder(SubDomain):
        def inside(self, x, on):
            return on and inside_cylinder(x, True)

    cylinder = Cylinder()
    mesh = mesh - cylinder

    # Boundary markers
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    boundaries.set_all(0)  # mark all subdomains as default (0)
    cylinder.mark(boundaries, 1)  # mark the cylinder boundary
    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

    # Inflow: x=-30
    inflow = MeshFunction("size_t", mesh, mesh.topology().dim()-1); inflow.set_all(0)
    inflow.sub(0).mark(boundaries, 2)  # mark the inlet boundary
    ds_inflow = Measure("ds", domain=mesh, subdomain_data=inflow)

    # Outflow: x=30
    outflow = MeshFunction("size_t", mesh, mesh.topology().dim()-1); outflow.set_all(0)
    outflow.sub(0).mark(boundaries, 3)  # mark the outlet boundary
    ds_outflow = Measure("ds", domain=mesh, subdomain_data=outflow)

    # Top and bottom boundaries: y=±30
    top_bottom = MeshFunction("size_t", mesh, mesh.topology().dim()-1); top_bottom.set_all(0)
    top_bottom.sub(0).mark(boundaries, 4)  # mark the top boundary
    ds_top_bottom = Measure("ds", domain=mesh, subdomain_data=top_bottom)

    # Cylinder boundaries: x²+y²≤0.25²
    cylinder_bound = MeshFunction("size_t", mesh, mesh.topology().dim()-1); cylinder_bound.set_all(0)
    cylinder_bound.sub(0).mark(boundaries, 5)  # mark the cylinder boundary

    # Variational form
    U = Constant((1.0, 0.0))
    u, v = TrialFunction(V), TestFunction(V)
    f = Constant((0.0, -9.81))

    a = (inner(grad(u), grad(v))*dx + U[0]*U[0]*(dot(n, grad(v)) + dot(grad(u), n))*ds
         + 1e-3*div(v)*div(u)*dx + 1e-2*(n[0]*v[1] - v[0]*n[1])*ds)
    L = dot(f, v)*dx

    # Initial condition (u₀=0, p₀=0)
    u0 = Expression(("0.0", "0.0"), degree=0)
    p0 = Constant(0.0)

    # Time-stepping
    t = 0.0
    dt = 1e-3
    T = 10.0

    def compute_drag(u):
        return sum([dot(U, u) * ds_inflow])

    while t < T:
        print("Time: %g" % t)

        # Solve for pressure and velocity
        p, u_t = solve(a == L, p, u0, [u, u_t], solver_parameters={"linear_solver": "mumps"})
        u_n = project(u + dt*u_t, V)
        u0.assign(u_n)

        # Compute drag coefficient
        C_d = compute_drag(u_n)

        print("C_d: %g" % C_d)

        t += dt

    # Save solution
    plot(mesh); plt.show()
    plot(u); plt.colorbar(); plt.title("Velocity (m/s)"); plt.show()
    plot(p); plt.colorbar(); plt.title("Pressure (Pa)"); plt.show()

if __name__ == "__main__":
    main()
