"""Mesh generation utilities.

Provides:
- mesh_plate_hole: Gmsh-based mesh of a plate with a central hole and ring.
"""
import gmsh


def mesh_plate_hole(length_x, length_y, hole_diameter, ring_diameter, mesh_size=None):
    """
    Create a 2D mesh of a rectangular plate with a central hole and ring.

    Parameters
    ----------
    length_x : float
        Plate dimension in the tension direction [mm].
    length_y : float
        Plate dimension in the transverse direction [mm].
    hole_diameter : float
        Diameter of the hole [mm].
    ring_diameter : float
        Diameter of the reinforcing ring around the hole [mm].
    mesh_size : float, optional
        Target element size. Defaults to min(length_x, length_y) / 10.

    Returns
    -------
    gmsh.model
        The Gmsh model (caller must finalize).
    """
    if hole_diameter >= min(length_x, length_y):
        raise ValueError(
            f"Hole diameter ({hole_diameter}) must be smaller than plate "
            f"dimensions ({length_x}\u00d7{length_y})."
        )
    if ring_diameter <= hole_diameter:
        raise ValueError(
            f"Ring diameter ({ring_diameter}) must be larger than hole "
            f"diameter ({hole_diameter})."
        )

    if mesh_size is None:
        mesh_size = min(length_x, length_y) / 10

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)

    gmsh.model.add('plate_with_hole')
    plate = gmsh.model.occ.addRectangle(0, 0, 0, length_x, length_y)
    hole = gmsh.model.occ.addDisk(
        length_x / 2, length_y / 2, 0, hole_diameter / 2, hole_diameter / 2
    )
    gmsh.model.occ.cut([(2, plate)], [(2, hole)])
    ring = gmsh.model.occ.addCircle(
        length_x / 2, length_y / 2, 0, ring_diameter / 2
    )
    gmsh.model.occ.fragment([(2, 1)], [(1, ring)])
    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(2, [1], 1)  # rubber
    gmsh.model.addPhysicalGroup(2, [2], 2)  # ring

    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
    gmsh.option.setNumber('Mesh.ElementOrder', 2)
    gmsh.model.mesh.generate(2)

    return gmsh.model
