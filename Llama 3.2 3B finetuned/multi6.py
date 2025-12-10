from dolfin import *
import mshr
import matplotlib.pyplot as plt

# 1. Define mesh and function space
mesh = mshr.RectangleMesh(mshr.Rectangle(Point(0, 0), Point(2.5, 0.41)), 64, 32)
V = VectorFunctionSpace(mesh, "P", 1)

# 2. Mark fluid domain (remove pole)
facets = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
tol = 1e-6
class Pole(SubDomain):
    def inside(self, x, on): return on and near(sqrt(x[0]**2 + x[1]**2), 0.05, tol)
Pole().mark(facets, 1)

# 3. Define fluid and solid domains
domain = mesh.subdomains()
fluid_domain = domain.copy(deepcopy=True) # copy to avoid modifying original
for i in range(len(domain)):
    if facets[i] == 1:
        fluid_domain[i] = False

# 4. Create function space for flag (remove pole)
flag_V = VectorFunctionSpace(mesh, "P", 1)

# 5. Define mesh functions for fluid and solid domains
fluid_meshfct = FunctionMeshFunction(domain, facets) # use domain as mesh function
solid_meshfct = FunctionMeshFunction(flag_V.submesh(), facets) # use flag submesh

# 6. Create boundary markers (remove pole)
bmark_f = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
bmark_s = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)

class Fluid(SubDomain):
    def inside(self, x, on): return on and fluid_meshfct(x) == 1
Fluid().mark(bmark_f, 1)

class Solid(SubDomain):
    def inside(self, x, on): return on and solid_meshfct(x) == 1
Solid().mark(bmark_s, 2)

# 7. Define boundary markers for fluid and solid domains
bmarks = [bmark_f, bmark_s]

# 8. Create mesh function for flag domain (remove pole)
flag_domain = MeshFunction("size_t", mesh, facets) # use domain as mesh function

class Flag(SubDomain):
    def inside(self, x, on): return on and flag_meshfct(x) == 1
Flag().mark(flag_domain, 3)

# 9. Create boundary markers for flag domain (remove pole)
flag_bmarks = [bmark_f, bmark_s]

# 10. Define fluid and solid domains
fluid_domain = MeshFunction("size_t", mesh, facets) # use domain as mesh function
solid_domain = MeshFunction("size_t", mesh, facets) # use domain as mesh function

class Fluid(SubDomain):
    def inside(self, x, on): return on and fluid_meshfct(x) == 1
Fluid().mark(fluid_domain, 1)

class Solid(SubDomain):
    def inside(self, x, on): return on and solid_meshfct(x) == 1
Solid().mark(solid_domain, 2)

# 11. Create boundary markers for fluid domain (remove pole)
fluid_bmarks = [bmark_f, bmark_s]

# 12. Define flag domain (remove pole)
flag_domain = MeshFunction("size_t", mesh, facets) # use domain as mesh function

class Flag(SubDomain):
    def inside(self, x, on): return on and flag_meshfct(x) == 1
Flag().mark(flag_domain, 3)

# 13. Create boundary markers for flag domain (remove pole)
flag_bmarks = [bmark_f, bmark_s]

# 14. Define mesh functions for fluid and solid domains
fluid_meshfct = FunctionMeshFunction(domain, facets) # use domain as mesh function
solid_meshfct = FunctionMeshFunction(flag_V.submesh(), facets) # use flag submesh

# 15. Create boundary markers (remove pole)
bmark_f = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
bmark_s = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)

class Fluid(SubDomain):
    def inside(self, x, on): return on and fluid_meshfct(x) == 1
Fluid().mark(bmark_f, 1)

class Solid(SubDomain):
    def inside(self, x, on): return on and solid_meshfct(x) == 1
Solid().mark(bmark_s, 2)

# 16. Define boundary markers for fluid domain (remove pole)
fluid_bmarks = [bmark_f, bmark_s]

# 17. Create mesh function for flag domain (remove pole)
flag_domain = MeshFunction("size_t", mesh, facets) # use domain as mesh function

class Flag(SubDomain):
    def inside(self, x, on): return on and flag_meshfct(x) == 1
Flag().mark(flag_domain, 3)

# 18. Create boundary markers for flag domain (remove pole)
flag_bmarks = [bmark_f, bmark_s]
