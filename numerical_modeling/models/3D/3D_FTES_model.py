#!/usr/bin/env python
# coding: utf-8

##############################################################################
################### 3D model for the full FTES experiment ####################
##############################################################################

import numpy as np
import io
from mpi4py import MPI
#import pyvista
import ufl
from ufl import Measure, FacetNormal
import matplotlib.pyplot as plt

import dolfinx
from dolfinx import fem, plot
from dolfinx.io import XDMFFile
from dolfinx.fem import FunctionSpace, VectorFunctionSpace, Constant, Function
from dolfinx.plot import create_vtk_mesh
from dolfinx.io.gmshio import read_from_msh

from petsc4py import PETSc
from petsc4py.PETSc import ScalarType


## Mesh reading

print("Reading mesh...")
mesh, cell_tags, facet_tags = read_from_msh("../../meshes/3D/3D_FTES_mesh.msh", MPI.COMM_WORLD, 0)
#print(np.unique(cell_tags.values))
#print(np.unique(facet_tags.values))
print("Reading is done.")


## Experiment Setup Parameters

# Geometrical parameters
Dpipe = 0.001          # inner diameter of the pipe is 1mm
# Experimental parameter
Qinj = 0.25e-6/60*1/4  # flow rate injected by the pump (*1/4 because we only model 1/4 of the block)


##############################################################################
############################# Part 1: fluid flow #############################
##############################################################################


print("PART 1: solve the steady state flow problem.")


## Submesh creation

#print(f"DOLFINx version: {dolfinx.__version__} based on GIT commit: {dolfinx.git_commit_hash} of https://github.com/FEniCS/dolfinx/")

print("Generating submesh...")
submesh, entity_map, _, _ = dolfinx.mesh.create_submesh(mesh, mesh.topology.dim, cell_tags.indices[(cell_tags.values==106)|(cell_tags.values==107)])

# method to transfer facet tags from parent mesh to submesh
# see post: https://fenicsproject.discourse.group/t/problem-transferring-facet-tags-to-submesh/11213
# CAREFUL: method works only for DOLFINx version v0.6.0
tdim = mesh.topology.dim
fdim = tdim - 1
c_to_f = mesh.topology.connectivity(tdim, fdim)
f_map = mesh.topology.index_map(fdim)
all_facets = f_map.size_local + f_map.num_ghosts
all_values = np.zeros(all_facets, dtype=np.int32)
all_values[facet_tags.indices] = facet_tags.values
#print(np.unique(all_values))

submesh.topology.create_entities(fdim)
subf_map = submesh.topology.index_map(fdim)
submesh.topology.create_connectivity(tdim, fdim)
c_to_f_sub = submesh.topology.connectivity(tdim, fdim)
num_sub_facets = subf_map.size_local + subf_map.size_global
sub_values = np.empty(num_sub_facets, dtype=np.int32)
for i, entity in enumerate(entity_map):
    parent_facets = c_to_f.links(entity)
    child_facets = c_to_f_sub.links(i)
    for child, parent in zip(child_facets, parent_facets):
        sub_values[child] = all_values[parent]
sub_meshtag = dolfinx.mesh.meshtags(submesh, submesh.topology.dim-1, np.arange(
    num_sub_facets, dtype=np.int32), sub_values)
print("Submesh is generated.")


## Output file for fluid flow problem

xdmf = XDMFFile(submesh.comm, "solution_3D_FTES_fluidPressure_model.xdmf", "w")
xdmf.write_mesh(submesh)


## Finite element function space for pressure field

U = FunctionSpace(submesh, ("CG", 4))  # Lagrange elements (degree 4)


## Trial and test functions

p, u = ufl.TrialFunction(U), ufl.TestFunction(U)


## Boundary conditions

# DIRICHLET: p=pext on side 2
pext = 0
boundary_dofs = fem.locate_dofs_topological(U, submesh.topology.dim-1, sub_meshtag.indices[sub_meshtag.values == 2])
bc_inj = fem.dirichletbc(ScalarType(pext), boundary_dofs, U)

bc_tot = [bc_inj]

# NEUMANN: q=Qinj/A on side 1
qin = -4*Qinj/(np.pi*Dpipe**2)


## Custom integration measures

# integrate over subdomains
dx = Measure("dx", domain=submesh)
# integrate over boundaries
ds = Measure("ds", domain=submesh, subdomain_data=sub_meshtag)
n = FacetNormal(submesh)


## Variational problem

a = ufl.dot(ufl.grad(p), ufl.grad(u)) * dx
L = - qin * u * ds(1)


## Linear solver

print("Solving steady state flow problem...")
problem = fem.petsc.LinearProblem(a, L, bcs=bc_tot, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})   # "pc_factor_mat_solver_type": "superlu_dist"
p_h = problem.solve()
xdmf.write_function(p_h)
xdmf.close()
print("Problem is solved.")


## Fluid flow (pressure gradient)

print("Computing fluid velocity field from pressure field...")
V = VectorFunctionSpace(submesh, ("CG", 3))
q_expr = fem.Expression(-ufl.grad(p_h), V.element.interpolation_points())
q_sub = fem.Function(V)
q_sub.interpolate(q_expr)
print("Computation is done.")

# save to visualize in paraview
xdmf = XDMFFile(submesh.comm, "solution_3D_FTES_fluidVelocity_model.xdmf", "w")
xdmf.write_mesh(submesh)
xdmf.write_function(q_sub)
xdmf.close()


##############################################################################
############################ Part 2: heat transfer ###########################
##############################################################################


print("PART 2: solve the transient heat transfer problem.")


## Output file for heat transfer problem

xdmf = XDMFFile(mesh.comm, "solution_3D_FTES_temperature_model.xdmf", "w")
xdmf.write_mesh(mesh)


## Temporal parameters

t = 0              # start time
T = 1000*8*3600.    # final time
num_steps = 200
dt = T / num_steps # time step size

Text = 20 # initial temperature in the system
Tinj = 70 # temperature of injected water


## Finite element function space for temperature field

W = FunctionSpace(mesh, ("CG", 1))  # Lagrange linear elements (degree 1)


## Initial conditions

T_n = Function(W)
T_n.name = "T_n"
T_n.x.array[:] = np.full(len(T_n.x.array), Text)

T_i = T_n.copy()
T_i.name = "T_i"


## Time-dependent output

T_h = T_n.copy()
T_h.name = "T_h"
xdmf.write_function(T_h, t)


## Trial and test functions

T, r = ufl.TrialFunction(W), ufl.TestFunction(W)


## Material properties

# for gabbro
rho_g = 3000                 # density of gabbro in kg/m³
c_g = 460                    # specific heat of gabbro in J/(kg*K)
cond_g = 2.15                # thermal conductivity of gabbro in W/(m·K)
print("Thermal diffusivity of gabbro: " + str(cond_g/(rho_g*c_g)))

# for water
rho_w = 997                  # density of water in kg/m³
c_w = 4182                   # specific heat of water in J/(kg*K)
cond_w = 0.598               # thermal conductivity of water in W/(m·K)
print("Thermal diffusivity of water: " + str(cond_w/(rho_w*c_w)))

# for metal (assume steel)
rho_m = 8000                 # density of metal in kg/m³
c_m = 420                    # specific heat of metal in J/(kg*K)
cond_m = 45                  # thermal conductivity of metal in W/(m·K)
print("Thermal diffusivity of metal: " + str(cond_m/(rho_m*c_m)))

# for epoxy resin
rho_e = 1100                 # density of epoxy resin in kg/m³
c_e = 1110                   # specific heat of epoxy resin in J/(kg*K)
cond_e = 0.14                # thermal conductivity of epoxy resin in W/(m·K)
print("Thermal diffusivity of epoxy resin: " + str(cond_e/(rho_e*c_e)))

# give those different properties to the domain
M = FunctionSpace(mesh, ("DG", 0))
metal_mask = (cell_tags.values == 103)
epoxy_mask = (cell_tags.values == 104)
rock_mask = (cell_tags.values == 105)
fluid_mask = (cell_tags.values == 106)|(cell_tags.values == 107)

rho = Function(M)
rho.x.array[metal_mask] = np.full(metal_mask.sum(), rho_m)
rho.x.array[epoxy_mask] = np.full(epoxy_mask.sum(), rho_e)
rho.x.array[rock_mask] = np.full(rock_mask.sum(), rho_g)
rho.x.array[fluid_mask] = np.full(fluid_mask.sum(), rho_w)

c = Function(M)
c.x.array[metal_mask] = np.full(metal_mask.sum(), c_m)
c.x.array[epoxy_mask] = np.full(epoxy_mask.sum(), c_e)
c.x.array[rock_mask] = np.full(rock_mask.sum(), c_g)
c.x.array[fluid_mask] = np.full(fluid_mask.sum(), c_w)

cond = Function(M)
cond.x.array[metal_mask] = np.full(metal_mask.sum(), cond_m)
cond.x.array[epoxy_mask] = np.full(epoxy_mask.sum(), cond_e)
cond.x.array[rock_mask] = np.full(rock_mask.sum(), cond_g)
cond.x.array[fluid_mask] = np.full(fluid_mask.sum(), cond_w)


## Fluid flow

print("Projecting fluid velocity field from submesh to parent mesh...")

# function to transfer data from submesh to parent mesh
# source: https://gist.github.com/jorgensd/9170f86a9e47d22b73f1f0598f038773
def transfer_submesh_data(u_parent: dolfinx.fem.Function, u_sub: dolfinx.fem.Function,
                          sub_to_parent_cells: np.ndarray, inverse: bool = False):
    """
    Transfer data between a function from the parent mesh and a function from the sub mesh.
    Both functions has to share the same element dof layout
    Args:
        u_parent: Function on parent mesh
        u_sub: Function on sub mesh
        sub_to_parent_cells: Map from sub mesh (local index) to parent mesh (local index)
        inverse: If true map from u_sub->u_parent else u_parent->u_sub
    """
    
    V_parent = u_parent.function_space
    V_sub = u_sub.function_space
    # FIXME: In C++ check elementlayout for equality
    if inverse:
        for i, cell in enumerate(sub_to_parent_cells):
            bs = V_parent.dofmap.bs
            bs_sub = V_sub.dofmap.bs
            assert(bs == bs_sub)
            parent_dofs = V_parent.dofmap.cell_dofs(cell)
            sub_dofs = V_sub.dofmap.cell_dofs(i)
            for p_dof, s_dof in zip(parent_dofs, sub_dofs):
                for j in range(bs):
                    u_parent.x.array[p_dof * bs + j] = u_sub.x.array[s_dof * bs + j]
    else:
        for i, cell in enumerate(sub_to_parent_cells):
            bs = V_parent.dofmap.bs
            bs_sub = V_sub.dofmap.bs
            assert(bs == bs_sub)
            parent_dofs = V_parent.dofmap.cell_dofs(cell)
            sub_dofs = V_sub.dofmap.cell_dofs(i)
            for p_dof, s_dof in zip(parent_dofs, sub_dofs):
                for j in range(bs):
                    u_sub.x.array[s_dof * bs + j] = u_parent.x.array[p_dof * bs + j]

# transfer solution of fluid flow to parent mesh
Q = VectorFunctionSpace(mesh, ("CG", 3))
q = Function(Q)
q.x.array[:] = 0
transfer_submesh_data(q, q_sub, entity_map, inverse=True)
q.x.scatter_forward()

print("Projection is done.")


## Boundary conditions

# DIRICHLET: T=Ti on side 1
boundary_dofs_inj = fem.locate_dofs_topological(W, mesh.topology.dim-1, facet_tags.indices[facet_tags.values == 1])
bc_inj = fem.dirichletbc(ScalarType(Tinj), boundary_dofs_inj, W)

bc_tot = [bc_inj]


## Custom integration measures

# integrate over subdomains
dx = Measure("dx", domain=mesh, subdomain_data=cell_tags)
# integrate over boundaries
ds = Measure("ds", domain=mesh, subdomain_data=facet_tags)
n = FacetNormal(mesh)


## Variational problem

F = rho * c * T * r * ufl.dx + dt * cond * ufl.dot(ufl.grad(T), ufl.grad(r)) * ufl.dx + dt * rho_w * c_w * ufl.dot(q, ufl.grad(T)) * r * ufl.dx \
    - (rho * c * T_n * r * ufl.dx)

a = ufl.lhs(F)
L = ufl.rhs(F)


## Linear algebra structures for the time dependent problem

bilinear_form = fem.form(a)
linear_form = fem.form(L)

A = fem.petsc.assemble_matrix(bilinear_form, bcs=bc_tot)
A.assemble()
b = fem.petsc.create_vector(linear_form)


## Linear solver

solver = PETSc.KSP().create(mesh.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)
#pc = solver.getPC()
#pc.setType(PETSc.PC.Type.LU)
#pc.setFactorSolverType("mumps")

print("Solving transient heat transfer problem...")

E_stored = [0]  # at the initial state no energy is stored
E_input_conduction = [0]
E_input_advection = [0]
E_output_advection = [0]
for i in range(num_steps):
    t += dt

    # Update the right hand side reusing the initial vector
    with b.localForm() as loc_b:
        loc_b.set(0)
    fem.petsc.assemble_vector(b, linear_form)
    
    # Apply Dirichlet boundary condition to the vector
    fem.petsc.apply_lifting(b, [bilinear_form], [bc_tot])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, bc_tot)

    # Solve linear problem
    solver.solve(b, T_h.vector)
    T_h.x.scatter_forward()

    # Update solution at previous time step (u_n)
    T_n.x.array[:] = T_h.x.array
    
    # Write solution to file
    xdmf.write_function(T_h, t)

    # Compute stored energy
    I1 = fem.form(rho * c * (T_h-T_i) * dx((103, 104, 105, 106, 107)))
    E_stored.append(fem.assemble_scalar(I1))
    #E_stored.append(MPI.COMM_WORLD.allreduce(fem.assemble_scalar(I1), op=MPI.SUM))
    
    # Compute input energy by conduction   
    I2 = fem.form(dt * cond * ufl.dot(ufl.grad(T_h), n) * ds(1))
    E_input_conduction.append(fem.assemble_scalar(I2))
    #E_input_conduction.append(MPI.COMM_WORLD.allreduce(fem.assemble_scalar(I2), op=MPI.SUM))
    
    # Compute input energy by advection
    I3 = fem.form(dt * rho * c * ufl.dot(q, -n) * (T_h-T_i) * ds(1))
    E_input_advection.append(fem.assemble_scalar(I3))
    #E_input_advection.append(MPI.COMM_WORLD.allreduce(fem.assemble_scalar(I3), op=MPI.SUM))
    
    # Compute output energy by advection
    I4 = fem.form(dt * rho * c * ufl.dot(q, n) * (T_h-T_i) * ds(2))
    E_output_advection.append(fem.assemble_scalar(I4))
    #E_output_advection.append(MPI.COMM_WORLD.allreduce(fem.assemble_scalar(I4), op=MPI.SUM))
    
    if i % 10 == 0:
        print("Step ", i, "/", num_steps)
        
xdmf.close()

print("Problem is solved.")

print("Verifying that the fluid flow rate leaving the fracture is equal to the influent flow rate at the wellhead...")
Itest = fem.form(ufl.dot(q, n) * ds(2))
print("Flow in = ", Qinj/4)
print("Flow out = ", fem.assemble_scalar(Itest))
print("Verification is done.")


## Post-processing of results (energy balance)

np.savez('arrays_post_processing', E_stored=np.array(E_stored), E_input_conduction=np.array(E_input_conduction).cumsum(), \
        E_input_advection=np.array(E_input_advection).cumsum(), E_output_advection=np.array(E_output_advection).cumsum())
