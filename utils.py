import numpy as np
import gmsh


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


import scipy.io
def read_mat_data(file):
    """
    Reads Matlab .mat file.

    Parameters:
    file (str): Path to the .mat file.

    Returns:
    dict: Dictionary containing the data from the .mat file.
    """
    data = scipy.io.loadmat(file)
    # Remove rows and columns with only 0 and replace 0 with NaN
    for field in data:
        data[field] = data[field][~np.all(data[field] == 0, axis=1)]
        data[field] = data[field][:, ~np.all(data[field] == 0, axis=0)]
        data[field][data[field] == 0] = np.nan
    return data


def mesh_plate_hole(length_x, length_y, hole_diameter, ring_diameter, mesh_size = None):
    """
    Creates a mesh of a plate with a hole in the center.

    Parameters:
    length_x (float): Dimension of the plate in the x direction.
    length_y (float): Dimension of the plate in the y direction.
    hole_diameter (float): Diameter of the hole.
    ring_diameter (float): Diameter of the ring.
    mesh_size (float, optional): Mesh size. Defaults to min(length_x, length_y) / 10.

    Returns:
    gmsh.model: The Gmsh model.
    """

    if mesh_size is None:
        mesh_size = min(length_x, length_y) / 10

    gmsh.initialize()

    # Reduce log level
    gmsh.option.setNumber("General.Verbosity", 0)

    gmsh.model.add('plate_with_hole')
    # Create the plate
    plate = gmsh.model.occ.addRectangle(0, 0, 0, length_x, length_y)
    # Create the hole
    hole = gmsh.model.occ.addDisk(length_x / 2, length_y / 2, 0, hole_diameter / 2, hole_diameter / 2)
    # Remove the hole from the plate
    plate_with_hole = gmsh.model.occ.cut([(2, plate)], [(2, hole)])
    # Create the ring
    ring = gmsh.model.occ.addCircle(length_x / 2, length_y / 2, 0, ring_diameter / 2)
    # Fragment the plate with the ring
    plate_with_hole_with_ring = gmsh.model.occ.fragment([(2, 1)], [(1, ring)])
    # Synchronize the model
    gmsh.model.occ.synchronize()
    # Add physical group for all entities of dimension 2
    gmsh.model.addPhysicalGroup(2, [1], 1)
    gmsh.model.addPhysicalGroup(2, [2], 2)
    # Create the mesh
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
    gmsh.option.setNumber('Mesh.ElementOrder', 2) # Quadratic elements
    gmsh.model.mesh.generate(2)
    return gmsh.model


    



from dolfinx.io.gmshio import model_to_mesh
from dolfinx import fem, mesh, plot, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import ufl
import numpy as np
import pyvista

# If we're running headless, we need to start a virtual display
pyvista.start_xvfb()

from mpi4py import MPI

comm = MPI.COMM_WORLD

def find_nearest(array, value):     
    array = np.asarray(array)    
    idx = (np.abs(array - value)).argmin()    
    return idx


class FEModel:

    def __init__(self, axial_length, transverse_length, thickness, hole_diameter, ring_diameter, SEF="Neo-Hookean", mesh_size=None):
        """
        Initializes a finite element simulation of a plate with a hole in the center.

        Parameters:
        axial_length (float): Length of the plate in the tension direction.
        transverse_length (float): Length of the plate in the transverse direction.
        thickness (float): Thickness of the plate.
        hole_diameter (float): Diameter of the hole.
        ring_diameter (float): Diameter of the ring.
        SEF (str, optional): Strain energy function. Defaults to "Neo-Hookean". Can be one of "Neo-Hookean" or "Mooney-Rivlin".
        mesh_size (float, optional): Mesh size.
        
        """

        # Create and import the mesh
        model = mesh_plate_hole(axial_length, transverse_length, hole_diameter, ring_diameter, mesh_size)
        domain, cell_tags, facet_tags = model_to_mesh(model, comm, rank=0, gdim=2)

        # Define the function space for the displacement field
        V = fem.functionspace(domain, ("Lagrange", 2, (domain.geometry.dim, )))

        # Define the boundary conditions
        def right(x):
            return np.isclose(x[0], axial_length)

        def left(x):
            return np.isclose(x[0], 0.)
        
        fdim = domain.topology.dim - 1 # Facet dimension

        # Locate dofs on the left
        facets_left = mesh.locate_entities_boundary(domain, fdim, left)
        u_dofs_left = fem.locate_dofs_topological(V, fdim, facets_left)
        # ux = uy = 0 on the left boundary
        bc_left = fem.dirichletbc(np.array([0, 0], dtype=default_scalar_type), u_dofs_left, V)

        # Locate dofs on the right
        facets_right = mesh.locate_entities_boundary(domain, fdim, right)
        ux_dofs_right = fem.locate_dofs_topological(V.sub(0), fdim, facets_right)
        dispBC = fem.Constant(domain, 0.)
        # ux = dispBC on the right boundary
        bc_ux_right = fem.dirichletbc(dispBC, ux_dofs_right, V.sub(0))
        # uy = 0 on the right boundary
        uy_dofs_right = fem.locate_dofs_topological(V.sub(1), fdim, facets_right)
        bc_uy_right = fem.dirichletbc(0., uy_dofs_right, V.sub(1))

        bcs = [bc_left, bc_ux_right, bc_uy_right]

        # To be used to compute the reaction force
        u_reac = fem.Function(V)
        ux_dofs_left = fem.locate_dofs_topological(V.sub(0), fdim, facets_left)
        bc_ux_left = fem.dirichletbc(fem.Constant(domain, 1.), ux_dofs_left, V.sub(0))
        fem.set_bc(u_reac.x.array, [bc_ux_left])

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
        axial_strain_expr = fem.Expression(E_GL[0,0], Vs.element.interpolation_points())
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


    def init_plot(self, movie_file=None):
        """
        Initializes the plot.

        Returns:
        pyvista.Plotter: The plotter.
        """

        # Extract the mesh
        topology, cells, geometry = plot.vtk_mesh(self.V)

        # Create reference and deformed function grids
        reference_function_grid = pyvista.UnstructuredGrid(topology, cells, geometry)
        deformed_function_grid = reference_function_grid.copy()

        # Add the displacement field
        deformed_function_grid["u"] = np.zeros((geometry.shape[0], 3))
        deformed_function_grid.set_active_vectors("u")

        # Add the axial strain field
        deformed_function_grid["axial_strain"] = np.zeros_like(self.axial_strain_fun.x.array)
        deformed_function_grid.set_active_scalars("axial_strain")

        # Create the plotter
        plotter = pyvista.Plotter(notebook=True)
        if movie_file is not None: plotter.open_movie(movie_file, framerate=10)
        plotter.add_mesh(deformed_function_grid, scalars = "axial_strain", show_edges=True, edge_opacity=0.2, lighting=False, scalar_bar_args={"title": "Axial strain [-]"})
        plotter.enable_parallel_projection()
        plotter.enable_terrain_style() # To disable mouse wheel zoom
        
        plotter.camera.tight(padding=0.1, adjust_render_window=False, view='xy')

        viewer = plotter.show(jupyter_backend="trame", return_viewer=True)

        self.reference_function_grid = reference_function_grid
        self.deformed_function_grid = deformed_function_grid
        self.plotter = plotter
        self.movie_file = movie_file
        self.saved_u = []
        self.saved_e = []

        return viewer
    

    def display_timestep(self, n):

        current_u = self.saved_u[n]
        current_e = self.saved_e[n]

        # Warp mesh
        self.deformed_function_grid.points[:, :2] = self.reference_function_grid.points[:, :2] + current_u.reshape(-1, 2)

        # Update the axial strain field
        self.deformed_function_grid.cell_data["axial_strain"][:] = current_e


    def solve(self, disp_values, show_progress=True, show_plot=True):
        """
        Solves the finite element problem for a series of displacement values.

        Parameters:
        disp_values (numpy.ndarray): Displacement values to apply.
        show_progress (bool, optional): If True, the progress will be displayed in the terminal. Defaults to True.
        show_plot (bool, optional): If True, the plot will be displayed. Defaults to True.

        Returns:
        numpy.ndarray: Model forces.
        """

        self.u.x.array[:] = 0
        self.u.x.scatter_forward()
        self.axial_strain_fun.x.array[:] = 0
        self.axial_strain_fun.x.scatter_forward()

        if self.plotter is not None and show_plot:
            self.saved_u = [np.zeros_like(self.u.x.array) for _ in range(len(disp_values))]
            self.saved_e = [np.zeros_like(self.axial_strain_fun.x.array) for _ in range(len(disp_values))]

        model_forces = np.nan*np.zeros(len(disp_values))

        for (n, d) in enumerate(disp_values):
            self.dispBC.value = d
            num_its, converged = self.solver.solve(self.u)
            assert(converged)
            self.u.x.scatter_forward()
            model_forces[n] = -fem.assemble_scalar(fem.form(ufl.action(self.F, self.u_reac)))*self.thickness
            if show_progress:
                print(f"Increment {n}: Displacement: {d:.2f} mm, Force: {model_forces[n]:.2f} N")

            if self.plotter is not None and show_plot:
                self.axial_strain_fun.interpolate(self.axial_strain_expr)
                self.saved_u[n][:] = self.u.x.array
                self.saved_e[n][:] = self.axial_strain_fun.x.array
                if self.movie_file is not None:
                    self.plotter.write_frame()


        if self.plotter is not None and show_plot:
            self.plotter.add_slider_widget(
                callback=lambda value: self.display_timestep(find_nearest(disp_values, value)),
                rng=(disp_values[0], disp_values[-1]),
                value=disp_values[-1],
                title="Displacement BC [mm]",
                fmt="%.1f",
                style="modern",
                interaction_event="always",
            )
            self.plotter.camera.tight(padding=0.1, adjust_render_window=False, view='xy')
            self.plotter.update_scalar_bar_range([np.min(self.saved_e[-1]), np.max(self.saved_e[-1])])
            self.plotter.update()

        return model_forces


    def get_fields_on_undeformed_mesh(self):
        """
        Returns the strain field on the undeformed mesh.

        Returns:
        pyvista.UnstructuredGrid: The axial strain field.
        """
        self.axial_strain_fun.interpolate(self.axial_strain_expr)
        self.reference_function_grid["axial_strain"] = self.axial_strain_fun.x.array
        self.reference_function_grid["u"] = self.u.x.array.reshape(-1, 2)[:,0]
        self.reference_function_grid["v"] = self.u.x.array.reshape(-1, 2)[:,1]
        return self.reference_function_grid
        



def plot_fields_side_by_side(exp_fields, exp_fieldname, num_fields, num_fieldname, scalar_bar_title, px_to_mm_scale=1., common_clim=None, image_file=None):
    """
    Plots two fields side by side.

    Parameters:
    exp_fields (dict): The experimental fields as a dictionary.
    exp_fieldname (str): Field name to extract from the experimental fields.
    num_fields (pyvista.UnstructuredGrid): The numerical fields as a pyvista mesh.
    num_fieldname (str): Field name to extract from the numerical fields.
    scalar_bar_title (str): Title for the scalar bar.
    px_to_mm_scale (float, optional): Scale factor to convert pixels to mm if the experimental fields are in pixels. Defaults to 1.
    common_clim (list, optional): Common color limits for the two fields. Defaults to mean +/- 3*std of the experimental field.
    image_file (str, optional): Output file for the figure. Defaults to None.

    """

    plotter = pyvista.Plotter(shape=(2, 1), notebook=True)

    if exp_fieldname not in exp_fields:
        raise ValueError(f"Field '{exp_fieldname}' not found in 'exp_fields'.")
    
    exp_field = exp_fields[exp_fieldname]*px_to_mm_scale

    if common_clim is None:
        common_clim = [np.nanmean(exp_field) - 3*np.nanstd(exp_field), np.nanmean(exp_field) + 3*np.nanstd(exp_field)]


    # Experimental field
    plotter.subplot(0, 0)
    imdata = pyvista.ImageData(dimensions=(exp_field.shape[1], exp_field.shape[0], 1))
    imdata.point_data["axial_strain"] = exp_field.ravel(order="C")
    plotter.add_mesh(imdata, scalars="axial_strain", lighting=False, clim=common_clim, nan_opacity=0, scalar_bar_args={"title": f"Experimental {scalar_bar_title}"})
    plotter.view_xy()
    plotter.enable_parallel_projection()
    plotter.enable_terrain_style() # To disable mouse wheel zoom
    plotter.camera.tight(adjust_render_window=False, view='xy')

    # Numerical field
    plotter.subplot(1, 0)
    plotter.add_mesh(num_fields, scalars=num_fieldname, lighting=False, clim=common_clim, scalar_bar_args={"title": f"FE {scalar_bar_title}"})
    plotter.view_xy()
    plotter.enable_parallel_projection()
    plotter.enable_terrain_style() # To disable mouse wheel zoom
    plotter.camera.tight(adjust_render_window=False, view='xy')

    if image_file is not None: plotter.screenshot(image_file)  

    viewer = plotter.show(jupyter_backend="trame", return_viewer=True)

    plotter.close()

    return viewer

    