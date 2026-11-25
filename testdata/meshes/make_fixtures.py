"""Generate the gmsh test meshes of slidesim (format 4.1 ASCII, coordinates in um).

Run from the repository root:

    uvx --from gmsh python testdata/meshes/make_fixtures.py

- `unit_square.msh`: the square [0, 1] x [0, 1] um, area `domain` (1), edges `left` (11),
  `right` (12), `top` (13), `bottom` (14).
- `tissue_small.msh`: the square [0, 200] x [0, 200] um with two circular holes of radius
  `HOLE_RADIUS` um at `HOLE_CENTERS`, named as slideviz (`mesh/build_mesh.py`) names them: area
  `tissue` (3), edges `vessel_wall` (11, the hole boundaries) and `section_edge` (12, the outer
  boundary). Mesh size about `TISSUE_SIZE` um.
"""

from pathlib import Path

import gmsh

HERE = Path(__file__).resolve().parent

HOLE_RADIUS = 20.0
HOLE_CENTERS = ((60.0, 100.0), (140.0, 100.0))
TISSUE_SIZE = 10.0
UNIT_SIZE = 0.25


def start(name: str, size: float) -> None:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.NumThreads", 1)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMin", size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", size)
    gmsh.model.add(name)


def finish(name: str) -> None:
    gmsh.model.mesh.generate(2)
    gmsh.write(str(HERE / f"{name}.msh"))
    gmsh.finalize()


def unit_square() -> None:
    start("unit_square", UNIT_SIZE)
    occ = gmsh.model.occ
    surface = occ.addRectangle(0.0, 0.0, 0.0, 1.0, 1.0)
    occ.synchronize()
    gmsh.model.addPhysicalGroup(2, [surface], 1, "domain")
    edges = {"left": [], "right": [], "top": [], "bottom": []}
    for _, curve in gmsh.model.getBoundary([(2, surface)], oriented=False):
        x, y, _ = occ.getCenterOfMass(1, curve)
        name = "left" if x < 1e-9 else "right" if x > 1 - 1e-9 else "bottom" if y < 1e-9 else "top"
        edges[name].append(curve)
    for tag, name in enumerate(("left", "right", "top", "bottom"), start=11):
        gmsh.model.addPhysicalGroup(1, edges[name], tag, name)
    finish("unit_square")


def tissue_small() -> None:
    start("tissue_small", TISSUE_SIZE)
    occ = gmsh.model.occ
    square = occ.addRectangle(0.0, 0.0, 0.0, 200.0, 200.0)
    holes = [occ.addDisk(x, y, 0.0, HOLE_RADIUS, HOLE_RADIUS) for x, y in HOLE_CENTERS]
    (tissue, *_), _ = occ.cut([(2, square)], [(2, h) for h in holes])
    occ.synchronize()
    surface = tissue[1]
    gmsh.model.addPhysicalGroup(2, [surface], 3, "tissue")
    walls, section = [], []
    for _, curve in gmsh.model.getBoundary([(2, surface)], oriented=False):
        x, y, _ = occ.getCenterOfMass(1, curve)
        on_hole = any(abs(x - cx) < HOLE_RADIUS and abs(y - cy) < HOLE_RADIUS for cx, cy in HOLE_CENTERS)
        (walls if on_hole else section).append(curve)
    gmsh.model.addPhysicalGroup(1, walls, 11, "vessel_wall")
    gmsh.model.addPhysicalGroup(1, section, 12, "section_edge")
    finish("tissue_small")


if __name__ == "__main__":
    unit_square()
    tissue_small()
