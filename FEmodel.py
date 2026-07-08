"""FEniCSx-based finite element solver for a plate with a hole.

Provides:
- FEModel: hyperelastic FE solver (Neo-Hookean / Mooney-Rivlin)
"""
import numpy as np

from dolfinx.io.gmshio import model_to_mesh
from dolfinx import fem, mesh, plot, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import ufl

from mpi4py import MPI

from mesher import mesh_plate_hole

_comm = None


def _get_comm():
    global _comm
    if _comm is None:
        _comm = MPI.COMM_WORLD
    return _comm


class FEModel:
    """
    Finite element model of a rectangular plate with a central hole and
    reinforcing ring, under uniaxial tension.

    Supports Neo-Hookean and Mooney-Rivlin hyperelastic constitutive models
    under plane-stress assumption.

    Parameters
    ----------
    axial_length : float
        Plate length in the tension direction [mm].
    transverse_length : float
        Plate length in the transverse direction [mm].
    thickness : float
        Out-of-plane thickness [mm].
    hole_diameter : float
        Diameter of the central hole [mm].
    ring_diameter : float
        Diameter of the reinforcing ring [mm].
    SEF : str
        Strain energy function: ``"Neo-Hookean"`` or ``"Mooney-Rivlin"``.
    mesh_size : float, optional
        Target element size [mm].
    """

    def __init__(self, axial_length, transverse_length, thickness,
                 hole_diameter, ring_diameter, SEF="Neo-Hookean",
                 mesh_size=None):
        for name, val in [("axial_length", axial_length),
                          ("transverse_length", transverse_length),
                          ("thickness", thickness),
                          ("hole_diameter", hole_diameter),
                          ("ring_diameter", ring_diameter)]:
            if val <= 0:
                raise ValueError(f"'{name}' must be positive, got {val}.")
        if SEF not in ("Neo-Hookean", "Mooney-Rivlin"):
            raise ValueError(
                f"SEF must be 'Neo-Hookean' or 'Mooney-Rivlin', got '{SEF}'."
            )

        gmsh_model = mesh_plate_hole(axial_length, transverse_length,
                                     hole_diameter, ring_diameter, mesh_size)
        domain, cell_tags, facet_tags = model_to_mesh(
            gmsh_model, _get_comm(), rank=0, gdim=2
        )

        V = fem.functionspace(
            domain, ("Lagrange", 2, (domain.geometry.dim,))
        )

        def right(x):
            return np.isclose(x[0], axial_length)

        def left(x):
            return np.isclose(x[0], 0.0)

        fdim = domain.topology.dim - 1

        facets_left = mesh.locate_entities_boundary(domain, fdim, left)
        u_dofs_left = fem.locate_dofs_topological(V, fdim, facets_left)
        bc_left = fem.dirichletbc(
            np.array([0, 0], dtype=default_scalar_type), u_dofs_left, V
        )

        facets_right = mesh.locate_entities_boundary(domain, fdim, right)
        ux_dofs_right = fem.locate_dofs_topological(V.sub(0), fdim, facets_right)
        dispBC = fem.Constant(domain, 0.0)
        bc_ux_right = fem.dirichletbc(dispBC, ux_dofs_right, V.sub(0))

        uy_dofs_right = fem.locate_dofs_topological(V.sub(1), fdim, facets_right)
        bc_uy_right = fem.dirichletbc(0.0, uy_dofs_right, V.sub(1))

        bcs = [bc_left, bc_ux_right, bc_uy_right]

        u_reac = fem.Function(V)
        ux_dofs_left = fem.locate_dofs_topological(V.sub(0), fdim, facets_left)
        bc_ux_left_reac = fem.dirichletbc(
            fem.Constant(domain, 1.0), ux_dofs_left, V.sub(0)
        )
        fem.set_bc(u_reac.x.array, [bc_ux_left_reac])

        u = fem.Function(V)
        v = ufl.TestFunction(V)

        cells_rubber = cell_tags.find(1)
        cells_ring = cell_tags.find(2)
        Q = fem.functionspace(domain, ("DG", 0))

        if SEF == "Neo-Hookean":
            C1 = fem.Function(Q)
            C1.x.array[cells_rubber] = np.full_like(cells_rubber, 0.0,
                                                     dtype=default_scalar_type)
            C1.x.array[cells_ring] = np.full_like(cells_ring, 0.0,
                                                   dtype=default_scalar_type)
            mat_params = {"C1": C1}
        elif SEF == "Mooney-Rivlin":
            C1 = fem.Function(Q)
            C2 = fem.Function(Q)
            C1.x.array[cells_rubber] = np.full_like(cells_rubber, 0.0,
                                                     dtype=default_scalar_type)
            C1.x.array[cells_ring] = np.full_like(cells_ring, 0.0,
                                                   dtype=default_scalar_type)
            C2.x.array[cells_rubber] = np.full_like(cells_rubber, 0.0,
                                                     dtype=default_scalar_type)
            C2.x.array[cells_ring] = np.full_like(cells_ring, 0.0,
                                                   dtype=default_scalar_type)
            mat_params = {"C1": C1, "C2": C2}

        def psi_NH(F):
            C = F.T * F
            C33 = 1 / ufl.det(C)
            I1 = ufl.tr(C) + C33
            return C1 * (I1 - 3)

        def psi_MR(F):
            C = F.T * F
            C33 = 1 / ufl.det(C)
            trC = ufl.tr(C) + C33
            I1 = trC
            trCsq = ufl.tr(C * C) + C33 ** 2
            I2 = 0.5 * (trC ** 2 - trCsq)
            return C1 * (I1 - 3) + C2 * (I2 - 3)

        psi = {"Neo-Hookean": psi_NH, "Mooney-Rivlin": psi_MR}[SEF]

        I = ufl.Identity(2)
        Fgrad = I + ufl.grad(u)
        Fvar = ufl.variable(Fgrad)
        P = ufl.diff(psi(Fvar), Fvar)
        E_GL = 0.5 * (Fgrad.T * Fgrad - I)

        dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 4})
        F_form = ufl.inner(P, ufl.grad(v)) * dx

        problem = NonlinearProblem(F_form, u, bcs)
        solver = NewtonSolver(_get_comm(), problem)
        solver.atol = 1e-8
        solver.rtol = 1e-8
        solver.convergence_criterion = "incremental"

        Vs = fem.functionspace(domain, ("DG", 0))
        axial_strain_expr = fem.Expression(
            E_GL[0, 0], Vs.element.interpolation_points()
        )
        axial_strain_fun = fem.Function(Vs)
        axial_strain_fun.interpolate(axial_strain_expr)

        self.solver = solver
        self.V = V
        self.u = u
        self.F_form = F_form
        self.E_GL = E_GL
        self.u_reac = u_reac
        self.dispBC = dispBC
        self.thickness = thickness
        self.domain = domain
        self.cells_rubber = cells_rubber
        self.cells_ring = cells_ring
        self.mat_params = mat_params
        self.SEF = SEF
        self.axial_strain_expr = axial_strain_expr
        self.axial_strain_fun = axial_strain_fun
        self.saved_u = []
        self.saved_e = []

    def set_material_parameters_NH(self, C1_rubber, C1_ring):
        """Set C₁ for rubber and ring (Neo-Hookean model)."""
        self.mat_params["C1"].x.array[self.cells_rubber] = np.full_like(
            self.cells_rubber, C1_rubber, dtype=default_scalar_type
        )
        self.mat_params["C1"].x.array[self.cells_ring] = np.full_like(
            self.cells_ring, C1_ring, dtype=default_scalar_type
        )

    def set_material_parameters_MR(self, C1_rubber, C1_ring, C2_rubber, C2_ring):
        """Set C₁, C₂ for rubber and ring (Mooney-Rivlin model)."""
        self.mat_params["C1"].x.array[self.cells_rubber] = np.full_like(
            self.cells_rubber, C1_rubber, dtype=default_scalar_type
        )
        self.mat_params["C1"].x.array[self.cells_ring] = np.full_like(
            self.cells_ring, C1_ring, dtype=default_scalar_type
        )
        self.mat_params["C2"].x.array[self.cells_rubber] = np.full_like(
            self.cells_rubber, C2_rubber, dtype=default_scalar_type
        )
        self.mat_params["C2"].x.array[self.cells_ring] = np.full_like(
            self.cells_ring, C2_ring, dtype=default_scalar_type
        )

    def solve(self, disp_values, show_progress=True):
        """
        Run the FE simulation.

        Returns
        -------
        model_forces : np.ndarray
        saved_u : list of np.ndarray
        saved_e : list of np.ndarray
        """
        disp_values = np.asarray(disp_values)

        self.u.x.array[:] = 0
        self.u.x.scatter_forward()
        self.axial_strain_fun.x.array[:] = 0
        self.axial_strain_fun.x.scatter_forward()

        n_steps = len(disp_values)
        saved_u = [np.zeros_like(self.u.x.array) for _ in range(n_steps)]
        saved_e = [np.zeros_like(self.axial_strain_fun.x.array)
                   for _ in range(n_steps)]
        model_forces = np.full(n_steps, np.nan)

        for n, d in enumerate(disp_values):
            self.dispBC.value = d
            num_its, converged = self.solver.solve(self.u)

            if not converged:
                print(
                    f"WARNING: Newton did not converge at inc {n} "
                    f"(disp = {d:.2f} mm)."
                )
                return model_forces[:n + 1], saved_u, saved_e

            self.u.x.scatter_forward()
            model_forces[n] = (
                -fem.assemble_scalar(
                    fem.form(ufl.action(self.F_form, self.u_reac))
                ) * self.thickness
            )

            if show_progress:
                print(
                    f"Increment {n}: "
                    f"Displacement: {d:.2f} mm, "
                    f"Force: {model_forces[n]:.2f} N"
                )

            self.axial_strain_fun.interpolate(self.axial_strain_expr)
            saved_u[n][:] = self.u.x.array
            saved_e[n][:] = self.axial_strain_fun.x.array

        self.saved_u = saved_u
        self.saved_e = saved_e
        return model_forces, saved_u, saved_e

    def get_reference_grid(self):
        """Return the undeformed FE mesh as (topology, cells, geometry) arrays."""
        return plot.vtk_mesh(self.V)

    def get_fields_on_undeformed_mesh(self):
        """Return the strain and displacement fields on the undeformed mesh.

        Returns
        -------
        topology, cells, geometry : arrays
            Mesh data from dolfinx.
        axial_strain : np.ndarray
            Cell-centered axial strain.
        u, v : np.ndarray
            Point-centered displacement components.
        """
        self.axial_strain_fun.interpolate(self.axial_strain_expr)
        topology, cells, geometry = plot.vtk_mesh(self.V)
        return (
            topology, cells, geometry,
            self.axial_strain_fun.x.array,
            self.u.x.array.reshape(-1, 2)[:, 0],
            self.u.x.array.reshape(-1, 2)[:, 1],
        )



