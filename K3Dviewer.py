import numpy as np
import k3d
import ipywidgets as widgets
from IPython.display import display


def _extract_linear_triangles(topology):
    """Extract linear triangle connectivity from VTK topology.

    ``topology`` is in VTK offset format:
    [6, v0..v5, 6, v0..v5, ...] for quadratic triangles.
    We take the first 3 corner nodes of each cell.
    """
    flat = topology.reshape(-1)
    stride = int(flat[0]) + 1  # nodes_per_cell + 1 (prefix)
    n_cells = flat.size // stride
    indexed = flat.reshape(n_cells, stride)[:, 1:4]  # skip prefix, keep 3 corners
    return indexed.astype(np.uint32)


class MeshViewer:
    """Interactive K3D time-series viewer for FE mesh results."""

    def __init__(self, model, disp_values):
        topology, _, geometry = model.get_reference_grid()
        self._tri = _extract_linear_triangles(topology)

        # 2D → 3D (pad z=0)
        self._verts = np.pad(geometry[:, :2].astype(np.float32),
                             ((0, 0), (0, 1)), mode='constant')
        self._geometry = geometry

        self._disp = np.asarray(disp_values)

        # --- K3D plot ---
        self.plot = k3d.plot(
            grid_visible=False, camera_auto_fit=False, menu_visibility=False,
            camera_no_rotate=True, background_color=0xFFFFFF,
        )

        self._k3d_mesh = k3d.mesh(
            vertices=self._verts, indices=self._tri,
            color_map=k3d.colormaps.matplotlib_color_maps.Viridis,
        )
        self._k3d_edges = k3d.mesh(
            vertices=self._verts, indices=self._tri,
            color=0x222222, wireframe=True,
        )
        self.plot += self._k3d_mesh
        self.plot += self._k3d_edges

        # --- UI ---
        self._label_force = widgets.Label(value="Force: Loading solver data...")

        self._step_slider = widgets.IntSlider(
            value=0, min=0, max=0, step=1,
            description='Timestep:', continuous_update=True,
            layout=widgets.Layout(width='400px'),
        )
        widgets.jslink((self.plot, 'time'), (self._step_slider, 'value'))

        title = widgets.HTML(
            "<div style='text-align:center;color:#333;font-size:14px;"
            "font-weight:bold;width:100%;'>Axial strain [-]</div>"
        )
        self.layout = widgets.VBox([
            title,
            widgets.HBox(
                [self._step_slider, self._label_force],
                layout=widgets.Layout(justify_content='center', alignment='center'),
            ),
            self.plot,
        ])

        fit_2d_camera(self.plot,
                      geometry[:, 0].min(), geometry[:, 0].max(),
                      geometry[:, 1].min(), geometry[:, 1].max())

        self.plot.observe(self._on_plot_time_change, names='time')

    def update_data(self, saved_u, saved_e, model_forces):
        """Push simulation frames to K3D."""
        self._forces = np.asarray(model_forces)
        n_steps = len(saved_u)
        n_nodes = self._geometry.shape[0]

        time_vertices = {}
        time_attributes = {}
        gmin, gmax = float('inf'), float('-inf')

        for idx in range(n_steps):
            t_str = str(idx)
            u = saved_u[idx].reshape(n_nodes, 2)

            # Deform 2D vertices
            verts_2d = self._geometry[:, :2] + u
            time_vertices[t_str] = np.pad(
                verts_2d.astype(np.float32), ((0, 0), (0, 1)), mode='constant'
            )

            # Per-cell strain → per-triangle (same mapping since cells == triangles)
            strain = saved_e[idx].astype(np.float32)
            time_attributes[t_str] = strain
            gmin = min(gmin, strain.min())
            gmax = max(gmax, strain.max())

        self._k3d_mesh.vertices = time_vertices
        self._k3d_edges.vertices = time_vertices
        self._k3d_mesh.triangles_attribute = time_attributes
        self._k3d_mesh.color_range = [float(gmin), float(gmax)]

        self._step_slider.max = n_steps - 1
        self.plot.time = float(n_steps - 1)

        fit_2d_camera(self.plot,
                      verts_2d[:, 0].min(), verts_2d[:, 0].max(),
                      verts_2d[:, 1].min(), verts_2d[:, 1].max())

    def _on_plot_time_change(self, change):
        if hasattr(self, '_forces'):
            idx = max(0, min(int(round(change['new'])), len(self._forces) - 1))
            self._label_force.value = (
                f"Disp: {self._disp[idx]:.2f} mm | "
                f"Force: {float(self._forces[idx]):.2f} N"
            )

    def display(self):
        display(self.layout)






def plot_fields_side_by_side(exp_fields, exp_fieldname, num_fields,
                             num_fieldname, scalar_bar_title,
                             px_to_mm_scale=1., common_clim=None):
    """Plot experimental (DIC) and numerical (FE) strain fields using K3D."""
    import numpy as np

    if exp_fieldname not in exp_fields:
        raise KeyError(
            f"Field '{exp_fieldname}' not found in exp_fields. "
            f"Available: {list(exp_fields.keys())}"
        )

    # --- Experimental (DIC) ---
    exp_field = exp_fields[exp_fieldname] * px_to_mm_scale
    ny, nx = exp_field.shape

    if common_clim is None:
        common_clim = [
            float(np.nanmean(exp_field) - 3 * np.nanstd(exp_field)),
            float(np.nanmean(exp_field) + 3 * np.nanstd(exp_field)),
        ]

    plot1 = k3d.plot(
        grid_visible=False, camera_auto_fit=False, camera_no_rotate=True,
        background_color=0xFFFFFF, menu_visibility=False, height=350
    )
    exp_surface = k3d.surface(
        heights=0.*exp_field.astype(np.float32),
        attribute=exp_field.astype(np.float32),
        bounds=[0, nx, 0, ny],
        color_map=k3d.colormaps.matplotlib_color_maps.Viridis,
        color_range=common_clim,
    )
    plot1 += exp_surface

    # --- Numerical (FE) ---
    # num_fields is the tuple from get_fields_on_undeformed_mesh:
    #   (topology, cells, geometry, axial_strain, u, v)
    topology, _, geometry, strain, u, v = num_fields
    tri = _extract_linear_triangles(topology)

    fe_verts = np.pad(geometry[:, :2].astype(np.float32),
                      ((0, 0), (0, 1)), mode='constant')

    plot2 = k3d.plot(
        grid_visible=False, camera_auto_fit=False, camera_no_rotate=True,
        background_color=0xFFFFFF, menu_visibility=False, height=350,
    )
    fe_mesh = k3d.mesh(
        vertices=fe_verts, indices=tri,
        triangles_attribute=strain.astype(np.float32),
        color_map=k3d.colormaps.matplotlib_color_maps.Viridis,
        color_range=common_clim,
    )
    plot2 += fe_mesh

    fit_2d_camera(plot1, 0, nx, 0, ny)
    fit_2d_camera(plot2,
                  geometry[:, 0].min(), geometry[:, 0].max(),
                  geometry[:, 1].min(), geometry[:, 1].max())

    title_exp = widgets.HTML(
        f"<div style='text-align:center;color:#333;font-size:14px;"
        f"font-weight:bold;width:100%;'>Experimental {scalar_bar_title}</div>"
    )
    title_num = widgets.HTML(
        f"<div style='text-align:center;color:#333;font-size:14px;"
        f"font-weight:bold;width:100%;'>FE {scalar_bar_title}</div>"
    )
    return widgets.VBox([
        widgets.VBox([title_exp, plot1]),
        widgets.VBox([title_num, plot2]),
    ])






def fit_2d_camera(plot_obj, x_min, x_max, y_min, y_max, margin=1., aspect_ratio=1.5):
    """Calculate and assign a perfect 2D camera viewport framing to a K3D plot.
    
    Parameters
    ----------
    plot_obj : k3d.Plot
        The target K3D plot instance to adjust.
    x_min, x_max, y_min, y_max : float
        The bounding box dimensions of the mesh or area to frame.
    margin : float, default 1.1
        Scale factor to add padding around the edges (1.1 = 10% padding).
    aspect_ratio : float, default 1.5
        Approximate width/height aspect ratio of your viewport container 
        to ensure wide models do not clip horizontally.
    """
    # 1. Find the geometric center point of the bounding box
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0

    # 2. Calculate physical dimensions
    width = x_max - x_min
    height = y_max - y_min
    
    # K3D's default perspective camera vertical FOV is 60 degrees.
    # tan(60° / 2) = tan(30°) ≈ 0.57735
    tan_half_fov = 0.57735
    
    # Strict geometric distance required to fit the height perfectly
    z_from_height = (height / 2.0) / tan_half_fov
    
    # Distance required to fit the width perfectly given the window aspect ratio
    z_from_width = (width / (2.0 * aspect_ratio)) / tan_half_fov
    
    # Select the dominant dimension to prevent any clipping edge-case
    best_z = max(z_from_height, z_from_width)
    
    # Fallback zoom distance if the mesh is empty or completely flat/point-like
    cam_z = best_z * margin if best_z > 0 else 150.0

    # 3. Update the K3D plot camera vector directly
    # Format: [cam_x, cam_y, cam_z, target_x, target_y, target_z, up_x, up_y, up_z]
    plot_obj.camera = [
        center_x, center_y, cam_z,  # Camera Position (Centered on X-Y, backed up on Z)
        center_x, center_y, 0.0,    # Target Look-At Point (Center of the mesh)
        0.0, 1.0, 0.0               # Up Vector (Enforces strict Y-is-Up 2D canvas)
    ]