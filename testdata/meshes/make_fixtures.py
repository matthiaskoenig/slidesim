"""Generate the gmsh test meshes of slidesim (format 4.1 ASCII, coordinates in um).

Run from the repository root:

    uvx --from gmsh python testdata/meshes/make_fixtures.py

- `unit_square.msh`: the square [0, 1] x [0, 1] um, area `domain` (1), edges `left` (11),
  `right` (12), `top` (13), `bottom` (14).
- `tissue_small.msh`: the square [0, 200] x [0, 200] um with two circular holes of radius
  `HOLE_RADIUS` um at `HOLE_CENTERS`, named as slideviz (`mesh/build_mesh.py`) names them: area
  `tissue` (3), edges `vessel_wall` (11, the hole boundaries) and `section_edge` (12, the outer
  boundary). Mesh size about `TISSUE_SIZE` um.
- `lobule_toy.msh`: the square [0, 400] x [0, 400] um with a circular hole of radius
  `LOBULE_HOLE_RADIUS` um at each of `LOBULE_PORTAL` and `LOBULE_CENTRAL` (near opposite corners):
  area `tissue` (3), edges `portal_wall` (11), `central_wall` (12) and `section_edge` (13, the outer
  boundary). Mesh size about `LOBULE_SIZE` um.
- `pinched.msh`: the rectangle [0, 4] x [0, 2] um with two triangular holes which touch at the
  point (2, 1), where the boundary of the domain touches itself: area `domain` (1), edges `left`
  (11), `right` (12), `rim` (13, top and bottom), `hole_wall` (14, both holes) and `inner_line`
  (15, the segment from (3.5, 0.5) to (3.5, 1.5) inside the domain, embedded in its mesh). Mesh
  size about `PINCHED_SIZE` um.

Written by hand, not by this script:

- `inverted_triangle.msh`: the square [0, 1] x [0, 1] um as two triangles, area `domain` (1).
  Triangle 1 lists its vertices counterclockwise, (0, 0), (1, 0), (1, 1); triangle 2 clockwise,
  (0, 0), (0, 1), (1, 1): an inverted element, which gmsh never writes. PETSc keeps the vertex
  order of the file, so its Jacobian determinant is negative.
- `unit_tetrahedron.msh`: the tetrahedron (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1) um, volume
  `domain` (1): a 3D mesh, for errors of code which supports 2D only.
"""

from pathlib import Path

import gmsh

HERE = Path(__file__).resolve().parent

HOLE_RADIUS = 20.0
HOLE_CENTERS = ((60.0, 100.0), (140.0, 100.0))
TISSUE_SIZE = 10.0
UNIT_SIZE = 0.25
LOBULE_HOLE_RADIUS = 20.0
LOBULE_PORTAL = (90.0, 90.0)
LOBULE_CENTRAL = (310.0, 310.0)
LOBULE_SIZE = 20.0
PINCHED_SIZE = 0.25


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


def lobule_toy() -> None:
    start("lobule_toy", LOBULE_SIZE)
    occ = gmsh.model.occ
    square = occ.addRectangle(0.0, 0.0, 0.0, 400.0, 400.0)
    centers = (LOBULE_PORTAL, LOBULE_CENTRAL)
    holes = [occ.addDisk(x, y, 0.0, LOBULE_HOLE_RADIUS, LOBULE_HOLE_RADIUS) for x, y in centers]
    (tissue, *_), _ = occ.cut([(2, square)], [(2, h) for h in holes])
    occ.synchronize()
    surface = tissue[1]
    gmsh.model.addPhysicalGroup(2, [surface], 3, "tissue")
    walls = {"portal_wall": [], "central_wall": [], "section_edge": []}
    for _, curve in gmsh.model.getBoundary([(2, surface)], oriented=False):
        x, y, _ = occ.getCenterOfMass(1, curve)
        name = "section_edge"
        for wall, (cx, cy) in zip(("portal_wall", "central_wall"), centers):
            if abs(x - cx) < LOBULE_HOLE_RADIUS and abs(y - cy) < LOBULE_HOLE_RADIUS:
                name = wall
        walls[name].append(curve)
    for tag, name in enumerate(("portal_wall", "central_wall", "section_edge"), start=11):
        gmsh.model.addPhysicalGroup(1, walls[name], tag, name)
    finish("lobule_toy")


def pinched() -> None:
    start("pinched", PINCHED_SIZE)
    geo = gmsh.model.geo

    def points(coordinates):
        return [geo.addPoint(x, y, 0.0, PINCHED_SIZE) for x, y in coordinates]

    def loop(corners):
        lines = [geo.addLine(a, b) for a, b in zip(corners, corners[1:] + corners[:1])]
        return lines, geo.addCurveLoop(lines)

    p0, p1, p2, p3 = points([(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)])
    outer, outer_loop = loop([p0, p1, p2, p3])
    (pinch,) = points([(2.0, 1.0)])
    a0, a1 = points([(1.0, 0.5), (1.0, 1.5)])
    b0, b1 = points([(3.0, 0.5), (3.0, 1.5)])
    hole_a, loop_a = loop([a0, pinch, a1])
    hole_b, loop_b = loop([b0, b1, pinch])
    surface = geo.addPlaneSurface([outer_loop, loop_a, loop_b])
    c0, c1 = points([(3.5, 0.5), (3.5, 1.5)])
    inner = geo.addLine(c0, c1)
    geo.synchronize()
    gmsh.model.mesh.embed(1, [inner], 2, surface)
    gmsh.model.addPhysicalGroup(2, [surface], 1, "domain")
    bottom, right, top, left = outer
    for tag, (name, curves) in enumerate(
        (
            ("left", [left]),
            ("right", [right]),
            ("rim", [top, bottom]),
            ("hole_wall", hole_a + hole_b),
            ("inner_line", [inner]),
        ),
        start=11,
    ):
        gmsh.model.addPhysicalGroup(1, curves, tag, name)
    finish("pinched")


if __name__ == "__main__":
    unit_square()
    tissue_small()
    lobule_toy()
    pinched()
