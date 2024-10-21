import numpy as np

import gmsh

import IPython.display as display


def commas_to_dots(x):
    return x.replace(',', '.').encode()



def extract_data_from_tensile_test(file, segment_number='all', tare=False):
    """
    Extracts data from a tensile test CSV file.

    Parameters:
    file (str): Path to the CSV file containing tensile test data.
    segment_number (str or int, optional): The segment number to extract. Defaults to 'all'.
        If 'all', data from all segments will be extracted.
    tare (bool, optional): If True, the data will be tared (i.e., the initial values will be subtracted). Defaults to True.

    Returns:
    tuple: A tuple containing three numpy arrays:
        - displacement (numpy.ndarray): Displacement data in mm.
        - force (numpy.ndarray): Force data in N.
        - time (numpy.ndarray): Time data in seconds.
    """
    # Read the CSV file
    data = np.genfromtxt((commas_to_dots(x) for x in open(file, encoding='iso-8859-1')), delimiter=';', skip_header=2)
    # First column is displacement in mm
    displacement = data[:, 0]
    # Second column is force in N
    force = data[:, 1]
    # Fourth column is time in s
    time = data[:, 3]
    # Sixth column is segment number
    segment = data[:, 5]
    # Give increasing numbers to segments
    prev_seg = segment[0]
    segment_id = 0
    for i in range(len(segment)):
        if segment[i] != prev_seg:
            prev_seg = segment[i]
            segment_id += 1
        segment[i] = segment_id
        
    if segment_number != 'all':
        mask = segment == segment_number
        displacement = displacement[mask]
        force = force[mask]
        time = time[mask]
    if tare:
        force -= force[0]
        time -= time[0]
        displacement -= displacement[0]
    return displacement, force, time




def mesh_plate_hole(width, height, hole_diameter, ring_diameter, mesh_size = None, outfile="Images/plate_with_hole.mp4"):
    """
    Creates a mesh of a plate with a hole in the center.

    Parameters:
    width (float): Width of the plate.
    height (float): Height of the plate.
    hole_diameter (float): Diameter of the hole.
    ring_diameter (float): Diameter of the ring.
    mesh_size (float, optional): Mesh size. Defaults to width / 10.

    Returns:
    gmsh.model: The Gmsh model.
    """

    if mesh_size is None:
        mesh_size = width / 10

    gmsh.initialize()

    # Reduce log level
    gmsh.option.setNumber("General.Verbosity", 0)

    gmsh.model.add('plate_with_hole')
    # Create the plate
    plate = gmsh.model.occ.addRectangle(0, 0, 0, width, height)
    # Create the hole
    hole = gmsh.model.occ.addDisk(width / 2, height / 2, 0, hole_diameter / 2, hole_diameter / 2)
    # Remove the hole from the plate
    plate_with_hole = gmsh.model.occ.cut([(2, plate)], [(2, hole)])
    # Create the ring
    ring = gmsh.model.occ.addCircle(width / 2, height / 2, 0, ring_diameter / 2)
    # Fragment the plate with the ring
    plate_with_hole_with_ring = gmsh.model.occ.fragment([(2, 1)], [(1, ring)])
    # Synchronize the model
    gmsh.model.occ.synchronize()
    # Add physical group for all entities of dimension 2
    gmsh.model.addPhysicalGroup(2, [1], 1)
    gmsh.model.addPhysicalGroup(2, [2], 2)
    # Create the mesh
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
    gmsh.option.setNumber('Mesh.ElementOrder', 2)
    gmsh.model.mesh.generate(2)
    return gmsh.model

import scipy.io
def read_mat_data(file, field):
    """
    Reads Matlab .mat file.

    Parameters:
    file (str): Path to the .mat file.
    field (str): Field to read from the .mat file.

    Returns:
    dict: Dictionary containing the data from the .mat file.
    """
    data = scipy.io.loadmat(file)
    if field not in data:
        raise ValueError(f"Field '{field}' not found in file '{file}'.")
    # Remove rows and columns with only 0
    data[field] = data[field][~np.all(data[field] == 0, axis=1)]
    data[field] = data[field][:, ~np.all(data[field] == 0, axis=0)]
    return data[field]
    



from dolfinx.io.gmshio import model_to_mesh
from dolfinx import fem, mesh, plot, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import ufl
import numpy as np
import pyvista

from mpi4py import MPI

comm = MPI.COMM_WORLD


class FEModel:

    def __init__(self, width, height, thickness, hole_diameter, ring_diameter, SEF="Neo-Hookean", mesh_size=None):
        """
        Initializes a finite element simulation of a plate with a hole in the center.

        Parameters:
        width (float): Width of the plate.
        height (float): Height of the plate.
        thickness (float): Thickness of the plate.
        hole_diameter (float): Diameter of the hole.
        ring_diameter (float): Diameter of the ring.
        SEF (str, optional): Strain energy function. Defaults to "Neo-Hookean". Can be one of "Neo-Hookean" or "Mooney-Rivlin".
        mesh_size (float, optional): Mesh size. Defaults to width / 10.
        
        """

        if mesh_size is None:
            mesh_size = width / 10

        # Create and import the mesh
        model = mesh_plate_hole(width, height, hole_diameter, ring_diameter, mesh_size)
        domain, cell_tags, facet_tags = model_to_mesh(model, comm, rank=0, gdim=2)

        # Define the function space for the displacement field
        V = fem.functionspace(domain, ("Lagrange", 2, (domain.geometry.dim, )))

        # Define the boundary conditions
        def top(x):
            return np.isclose(x[1], height)

        def bottom(x):
            return np.isclose(x[1], 0.)
        
        fdim = domain.topology.dim - 1 # Facet dimension

        # Locate dofs on the bottom boundary
        bot_facets = mesh.locate_entities_boundary(domain, fdim, bottom)
        u_dofs_bot = fem.locate_dofs_topological(V, fdim, bot_facets)
        # ux = uy = 0 on the bottom boundary
        bc_bot = fem.dirichletbc(np.array([0, 0], dtype=default_scalar_type), u_dofs_bot, V)

        # Locate dofs on the top boundary
        facets_top = mesh.locate_entities_boundary(domain, fdim, top)
        uy_dofs_top = fem.locate_dofs_topological(V.sub(1), fdim, facets_top)
        dispBC = fem.Constant(domain, 0.)
        # uy = dispBC on the top boundary
        bc_uy_top = fem.dirichletbc(dispBC, uy_dofs_top, V.sub(1))
        # ux = 0 on the top boundary
        ux_dofs_top = fem.locate_dofs_topological(V.sub(0), fdim, facets_top)
        bc_ux_top = fem.dirichletbc(0., ux_dofs_top, V.sub(0))

        bcs = [bc_bot, bc_uy_top, bc_ux_top]

        # To be used to compute the reaction force
        u_reac = fem.Function(V)
        uy_dofs_bot = fem.locate_dofs_topological(V.sub(1), fdim, bot_facets)
        bc_uy_bot = fem.dirichletbc(fem.Constant(domain, 1.), uy_dofs_bot, V.sub(1))
        fem.set_bc(u_reac.x.array, [bc_uy_bot])

        # Displacement field and test function
        u = fem.Function(V)
        v = ufl.TestFunction(V)
        
        # Define the material parameter field
        cells_1 = cell_tags.find(1) # Corresponds to the rubber
        cells_2 = cell_tags.find(2) # Corresponds to the ring

        Q = fem.functionspace(domain, ("DG", 0))
        if SEF == "Neo-Hookean":
            C1 = fem.Function(Q)
            C1.x.array[cells_1] = np.full_like(cells_1, 0, dtype=default_scalar_type)
            C1.x.array[cells_2] = np.full_like(cells_2, 0, dtype=default_scalar_type)
            mat_params = {"C1": C1}
        elif SEF == "Mooney-Rivlin":
            C1 = fem.Function(Q)
            C2 = fem.Function(Q)
            C1.x.array[cells_1] = np.full_like(cells_1, 0, dtype=default_scalar_type)
            C1.x.array[cells_2] = np.full_like(cells_2, 0, dtype=default_scalar_type)
            C2.x.array[cells_1] = np.full_like(cells_1, 0, dtype=default_scalar_type)
            C2.x.array[cells_2] = np.full_like(cells_2, 0, dtype=default_scalar_type)
            mat_params = {"C1": C1, "C2": C2}

        # Neo-Hookean strain energy density function
        def psi_NH(F):
            C = F.T * F
            C33 = 1/ufl.det(C) # Plane stress
            I1 = ufl.tr(C) + C33
            return C1*(I1-3) 
        
        # Mooney-Rivlin strain energy density function
        def psi_MR(F):
            C = F.T * F
            C33 = 1/ufl.det(C) # Plane stress
            trC = ufl.tr(C) + C33
            I1 = trC
            trCsquared = ufl.tr(C*C) + C33**2
            I2 = 0.5*(trC**2 - trCsquared)
            return C1*(I1-3) + C2*(I2-3)
        
        if SEF == "Neo-Hookean":
            psi = psi_NH
        elif SEF == "Mooney-Rivlin":
            psi = psi_MR
        else:
            raise ValueError(f"Strain energy function '{SEF}' not recognized.")
        
        I = ufl.Identity(2)
        Fgrad = I + ufl.grad(u)
        Fvar = ufl.variable(Fgrad)
        P = ufl.diff(psi(Fvar), Fvar) # First Piola-Kirchhoff stress = d(psi)/dF

        # Green-Lagrange strain tensor
        E_GL = 0.5*(Fgrad.T * Fgrad - I)

        dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 4})

        # Define the nonlinear form
        F = ufl.inner(P, ufl.grad(v))*dx

        # Create the nonlinear problem and associated Newton solver
        problem = NonlinearProblem(F, u, bcs)
        solver = NewtonSolver(comm, problem)

        # Set Newton solver options
        solver.atol = 1e-8
        solver.rtol = 1e-8
        solver.convergence_criterion = "incremental"

        # Compute the axial strain field
        Vs = fem.functionspace(domain, ("DG", 0))
        axial_strain_expr = fem.Expression(E_GL[1,1], Vs.element.interpolation_points())
        axial_strain_fun = fem.Function(Vs)
        axial_strain_fun.interpolate(axial_strain_expr)

        # Store the variables for later use
        self.solver = solver
        self.V = V
        self.u = u
        self.F = F
        self.E_GL = E_GL
        self.u_reac = u_reac
        self.dispBC = dispBC    
        self.thickness = thickness  
        self.domain = domain
        self.cells_1 = cells_1
        self.cells_2 = cells_2
        self.mat_params = mat_params
        self.axial_strain_expr = axial_strain_expr
        self.axial_strain_fun = axial_strain_fun
        
        self.plotter = None

    def set_material_parameters_NH(self, C1_rubber, C1_ring):
        """
        Sets the material parameters for the rubber and the ring for the Neo-Hookean model.

        Parameters:
        C1_rubber (float): Material parameter for the rubber.
        C1_ring (float): Material parameter for the ring.
        """
        self.mat_params["C1"].x.array[self.cells_1] = np.full_like(self.cells_1, C1_rubber, dtype=default_scalar_type)
        self.mat_params["C1"].x.array[self.cells_2] = np.full_like(self.cells_2, C1_ring, dtype=default_scalar_type)
        self.reset_fe_model()

    def set_material_parameters_MR(self, C1_rubber, C1_ring, C2_rubber, C2_ring):
        """
        Sets the material parameters for the rubber and the ring for the Mooney-Rivlin model.

        Parameters:
        C1_rubber (float): Material parameter for the rubber.
        C1_ring (float): Material parameter for the ring.
        C2_rubber (float): Material parameter for the rubber.
        C2_ring (float): Material parameter for the ring.
        """
        self.mat_params["C1"].x.array[self.cells_1] = np.full_like(self.cells_1, C1_rubber, dtype=default_scalar_type)
        self.mat_params["C1"].x.array[self.cells_2] = np.full_like(self.cells_2, C1_ring, dtype=default_scalar_type)
        self.mat_params["C2"].x.array[self.cells_1] = np.full_like(self.cells_1, C2_rubber, dtype=default_scalar_type)
        self.mat_params["C2"].x.array[self.cells_2] = np.full_like(self.cells_2, C2_ring, dtype=default_scalar_type)
        self.reset_fe_model()
        

    def reset_fe_model(self):
        """
        Resets the finite element model to the initial state.
        """
        self.u.x.array[:] = 0
        self.u.x.scatter_forward()
        self.axial_strain_fun.interpolate(self.axial_strain_expr)


    def init_plot(self, movie_file=None):
        """
        Initializes the plot.

        Returns:
        pyvista.Plotter: The plotter.
        """

        # Extract the mesh
        topology, cells, geometry = plot.vtk_mesh(self.V)
        reference_function_grid = pyvista.UnstructuredGrid(topology, cells, geometry)

        # Add the displacement field
        values = np.zeros((geometry.shape[0], 3))
        values[:, :2] = self.u.x.array.reshape(-1, 2)
        reference_function_grid["u"] = values

        # Warp mesh by displacement
        deformed_function_grid = reference_function_grid.warp_by_vector("u", factor=1)

        # Add the axial strain field
        deformed_function_grid.set_active_vectors("u")
        deformed_function_grid["axial_strain"] = self.axial_strain_fun.x.array
        deformed_function_grid.set_active_scalars("axial_strain")

        # Create the plotter
        plotter = pyvista.Plotter(notebook=True)
        if movie_file is not None: plotter.open_movie(movie_file, framerate=10)
        plotter.add_mesh(deformed_function_grid, scalars = "axial_strain", show_edges=True, edge_opacity=0.2, lighting=False, scalar_bar_args={"title": "Axial strain [-]"})
        plotter.enable_parallel_projection()
        plotter.enable_2d_style()
        plotter.view_yx()

        viewer = plotter.show(jupyter_backend="trame", return_viewer=True)

        self.reference_function_grid = reference_function_grid
        self.deformed_function_grid = deformed_function_grid
        self.plotter = plotter
        self.movie_file = movie_file

        return viewer
    
    def update_plot(self):
        """
        Updates the plot.
        """

        # Update the displacement field
        self.reference_function_grid["u"][:, :2] = self.u.x.array.reshape(-1, 2)
        self.axial_strain_fun.interpolate(self.axial_strain_expr)

        # Warp mesh by displacement
        self.deformed_function_grid.points[:, :] = self.reference_function_grid.warp_by_vector(factor=1).points

        # Update the axial strain field
        self.axial_strain_fun.interpolate(self.axial_strain_expr)
        self.deformed_function_grid.cell_data["axial_strain"][:] = self.axial_strain_fun.x.array

        # Update the plot limits
        self.plotter.view_yx()

        self.plotter.update()

        if self.movie_file is not None:
            self.plotter.write_frame()


    def solve(self, disp_values, show_progress=True, update_plot=True):
        """
        Solves the finite element problem for a series of displacement values.

        Parameters:
        disp_values (numpy.ndarray): Displacement values to apply.

        Returns:
        numpy.ndarray: Model forces.
        """

        model_forces = np.nan*np.zeros(len(disp_values))

        for (n, d) in enumerate(disp_values):
            self.dispBC.value = d
            num_its, converged = self.solver.solve(self.u)
            assert(converged)
            self.u.x.scatter_forward()
            model_forces[n] = -fem.assemble_scalar(fem.form(ufl.action(self.F, self.u_reac)))*self.thickness
            if show_progress:
                print(f"Increment {n}: Displacement: {d:.2f} mm, Force: {model_forces[n]:.2f} N")

            if update_plot and self.plotter is not None:
                self.update_plot()

        return model_forces


    def get_strain_field_on_undeformed_mesh(self):
        """
        Returns the strain field on the undeformed mesh.

        Returns:
        pyvista.UnstructuredGrid: The axial strain field.
        """
        self.axial_strain_fun.interpolate(self.axial_strain_expr)
        self.reference_function_grid["axial_strain"] = self.axial_strain_fun.x.array
        return self.reference_function_grid
        



def plot_fields_side_by_side(exp_field, num_field, image_file=None):
    """
    Plots two fields side by side.

    Parameters:
    exp_field (numpy.ndarray): The experimental field as a 2D numpy array.
    num_field (pyvista.UnstructuredGrid): The numerical field.
    image_file (str, optional): Output file for the figure. Defaults to None.

    """

    plotter = pyvista.Plotter(shape=(2, 1), notebook=True)

    common_clim = [np.min(exp_field), 0.5*np.max(exp_field)]

    # Experimental field
    plotter.subplot(0, 0)
    imdata = pyvista.ImageData(dimensions=(exp_field.shape[1], exp_field.shape[0], 1))
    imdata.point_data["axial_strain"] = exp_field.ravel(order="C")
    plotter.add_mesh(imdata, scalars="axial_strain", lighting=False, clim=common_clim, scalar_bar_args={"title": "Experimental axial strain [-]"})
    plotter.view_xy()
    # Temp fix for zoom not working
    plotter.camera.position = (plotter.camera.position[0], plotter.camera.position[1], plotter.camera.position[2]/2)

    # Numerical field
    plotter.subplot(1, 0)
    num_field.rotate_z(-90, inplace=True)
    plotter.add_mesh(num_field, scalars="axial_strain", lighting=False, clim=common_clim, scalar_bar_args={"title": "FE axial strain [-]"})
    plotter.view_xy()
    # Temp fix for zoom not working
    plotter.camera.position = (plotter.camera.position[0], plotter.camera.position[1], plotter.camera.position[2]/2)

    if image_file is not None: plotter.screenshot(image_file)  

    viewer = plotter.show(jupyter_backend="trame", return_viewer=True)

    plotter.close()

    return viewer

    