# Meshing

A scene gives its mesh in one of three ways (`mesh.format` of
[`scene/1`](../contracts.md#mesh-sources)): a gmsh file, which slidesim reads, or tagged polygons or
parametric lobules, which slidesim meshes itself when a run loads the scene. The mesher (crate
`mesher`) turns tagged polygons into a graded triangle mesh with the tags: a constrained Delaunay
triangulation (`spade`), adapted to a size field by Mmg through PETSc's `DMAdaptMetric`, then
improved locally. It is plain Rust but for Mmg, knows no tag names, and gives the same mesh, bit for
bit, for the same input on one target.

## In a run

Process 0 meshes the polygons or the lobules alone and tells the other processes the outcome.
The mesh becomes a DMPlex held by process 0, with one label per tag on every process, named
after the tag and with its value on the triangles of an area tag or the edges of an edge tag:
what PETSc makes of the physical groups of a gmsh file. From there every source takes the same
way: vertices where the boundary touches itself are split, PT-Scotch partitions the mesh, PETSc
distributes it and refines it `mesh.refinements` times. The coarse mesh, where the cells live,
is the mesh of the mesher.

A run prints the quality of a generated mesh in one line:

```text
mesh: 5902 triangles, 3093 vertices, min angle 25.46 deg, SICN min 0.641 mean 0.960, edge length / size 0.51 to 1.58
```

`slidesim mesh` writes the mesh of a scene without a run ([usage](../usage.md#meshes)). An input
the mesher cannot mesh is an error on every process which names the polygons file and the place
in um, for example (the path shortened; the message has the absolute path of the file)
`cannot mesh the polygons /data/field/tissue.geojson: rings or edges cross at (50, 50) um
(borders shared by areas must match exactly)`.

## The size field

The target edge length \(h(x)\) in um follows the distance \(d\) of \(x\) to the nearest segment of
the wall tags (`walls` of the `polygons` source; the centre and corner walls of the lobules):

$$
h(d) = \begin{cases}
  h_\min & d \le d_\min \\
  h_\min + (h_\max - h_\min) \dfrac{d - d_\min}{d_\max - d_\min} & d_\min < d < d_\max \\
  h_\max & d \ge d_\max
\end{cases}
$$

as gmsh's Threshold field, \(h_\max\) everywhere without walls. Several such fields combine by their
minimum. Narrow parts of the domain cap it: at a point of the boundary whose nearest segment across
(not joined to it along the boundary within twice the distance) is at the distance \(w\), the size
is at most \(0.4\,w\) (but not below \(h_\min / 4\)), growing by 0.3 per um away from that point, so
that the elements fit between the frozen boundaries of a neck. A chain across a neck (an interface
between two areas) joins its two sides within its own length, so next to such a chain the neck is no
gap; the cap holds where the way along the boundary is longer. Short boundary segments cap it too:
at both ends of a segment of length \(l\) the size is at most \(1.5\,l\) (but not below \(h_\min /
4\)), growing by 0.3 per um away from them. Every input vertex is kept, so a segment much shorter
than the size (a pixel step of a mask, 1.77 um under 5 um elements) is a frozen edge; with the cap
the triangles beside it take its length (an isosceles triangle on it has SICN 0.89) instead of
reaching 5 um inward.

## The steps

1. **Pieces.** The rings of the areas and the edge chains become segments, matched by exact
   coordinates; a segment with an input vertex on it is split there, so a border shared by two
   areas becomes the same pieces whatever vertices each side has. Pieces which cross are an
   error naming the crossing.
2. **Boundary.** Every piece is split to the size field, in equal steps of \(\int ds / h\). This is
   the final boundary: the steps after it keep every boundary and interface vertex, so every area
   keeps its polygons exactly and every tagged chain its length.
3. **Constrained Delaunay triangulation.** `spade` triangulates the points with every sub-edge as
   a constraint. The triangles are flood filled into regions across the edges which are no
   constraints, and each region is classified by its winding number around every polygon: inside
   exactly one polygon it belongs to its area, inside none it is outside, inside two the areas
   overlap (an error). Every piece between an area and the outside needs an edge tag. The
   triangles are refined until their longest edge is at most twice the size at their centroid,
   so that the metric interpolated on them carries the size field.
4. **Metric adaptation.** Mmg remeshes the triangulation to the isotropic metric
   \(M(x) = h(x)^{-2} I\) at the vertices (`DMAdaptMetric` on one process), with the boundary and
   the interfaces frozen (`no_surf`): Mmg moves, inserts and removes interior vertices only. The
   cells carry the region of step 3 as their reference, so a tagged chain between two parts of
   one area survives too.
5. **Improvement.** Mmg aims at edge lengths, not angles, and next to frozen walls leaves a few
   elements below the bounds. Around the elements which score below a margin above the bounds
   (`GOOD`: an angle of 23 degrees, a SICN of 0.46, edge ratios within [0.46, 1.74]), free
   vertices move towards the centroid of their neighbours, collapse onto a neighbour, or have
   their edges flipped, each change applied only if the worst element it makes is better than
   the worst it replaces, until the elements around a vertex reach that margin or a budget of
   operations is spent. The quality bounds below rest on this step: nothing guarantees that it
   lifts every element Mmg leaves below them, so the quality sweeps of the tests check it (rings
   0 and 1 in the full test run, rings 2 to 4 in the release step `quality_sweep_heavy` of
   `scripts/check.sh`, before every push).
6. **Quality.** The smallest angle, the SICN of every triangle (\(4\sqrt{3} A / (l_1^2 + l_2^2 +
   l_3^2)\), 1 for an equilateral triangle, 0 for a degenerate one) and the ratio of every edge
   length to the size at its midpoint. It is measured and reported, not enforced. The tests see
   a smallest angle of 20 degrees, a SICN of 0.4 and edge ratios within [0.4, 2.0] when the size
   field is one the boundary splitting resolves (no jump within less than an element), the gaps
   are not narrower than \(h_\min / 1.6\), the input segments not shorter than \(h_\min / 4\)
   (where both caps reach their floor) and the boundary has no sharp corners (the triangles at a
   corner of angle \(t\) have angles of at most \(t\) there). The lobules of any valid scene
   (tested over rings 0 to 4, wall sizes 2 to 20 um, jitter 0 to 0.25), polygonized circles and
   pixel steps of masks with pixels of at least \(h_\min / 4\) are such inputs; the real-slide
   fields are measured in the [milestone 4 meshing results](../results/2026-11-m4-meshing.md).

## Ideal lobules

The `lobules` source builds the polygons itself. The veins lie on a hexagonal lattice of spacing
\(\sqrt{3} R\) (\(R\) = `lobule_radius_um`, the circumradius of a lobule), ring by ring around the
central one, each displaced by a seeded uniform point of the disc of radius `jitter` \(R\). The
tissue is the union of the Voronoi cells of the veins within `rings` rings; the corners of the
cells, where three lobules meet, carry the portal fields. The holes are circles around the veins
(`centre`) and the corners (`corner`), polygonized with \(\max(12, \lceil 2 \pi r / h_w \rceil)\)
vertices on the full circle (\(h_w\) = `size.wall_um`); a corner hole on the outer border is cut
off by it. The edge tags are `centre.tag` and `corner.tag` on the hole walls and `outer_tag` on
the outer border, the area tag `area_tag`. The size field is `wall_um` up to `2 wall_um` from the
hole walls, linear to `interior_um` at `10 interior_um`, with the size at the wall of a small
hole lowered to the length of its polygon's segments.

### Zonation

The zonation map of the lobules (`zonation.map`) is the relative position of a point between the
two kinds of walls:

$$
p(x) = \frac{d_p(x)}{d_p(x) + d_c(x)}
$$

with \(d_p\) the distance to the nearest corner wall and \(d_c\) the distance to the nearest centre
wall (the circles, both 0 inside a hole): 0 on the corner (portal) walls, 1 on the centre
(central) walls, and monotone on the line between them. Each process computes it at the pixel
centres of a raster of `zonation.um_per_px` over the bounding box of the lobules (the first pixel
centred at its lower corner), for the window over its own cells only, and samples it bilinearly
like a map read from a file; its gradient gives the direction of the permeability of a perfusion
study.

## Verification

- The mesher: tags preserved (every area and tagged edge of the input in the mesh, chains
  between parts of one area included), the size field honoured within the edge ratio bounds,
  the quality bounds, the same mesh twice, and errors naming the place for crossings, overlaps,
  gaps between areas and untagged boundaries (`crates/mesher/tests`).
- The lobules: area and wall lengths of one lobule and of a ring of 7 against the analytic
  geometry, the zonation 0 on the corner walls, 1 on the centre walls and monotone between them,
  seeded jitter, and the quality bounds over a sweep of the parameters.
- In runs: the mesh of a gmsh file through the mesher's mesh and through the file are the same
  (`from_tagged_matches_load_of_the_same_mesh`); the lobule scene loads with its tags and area on
  1 to 3 processes (`lobule_scene_loads`); an error of the mesher on process 0 ends every process
  with the error (`mesher_failure_on_process_0_stops_all`); and the coupled tests run on the
  generated lobule: conservation, agreement of 1, 2 and 3 processes, and the order of the Strang
  splitting ([Cell models](cells.md#verification)).
