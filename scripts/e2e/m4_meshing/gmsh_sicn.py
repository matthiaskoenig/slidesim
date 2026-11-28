"""gmsh's minSICN of the triangles of a .msh file, unrounded."""

import sys

import gmsh
import numpy as np

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.open(sys.argv[1])
_, elements, _ = gmsh.model.mesh.getElements(2)
quality = np.asarray(gmsh.model.mesh.getElementQualities(np.concatenate(elements), "minSICN"))
gmsh.finalize()
print(repr(float(quality.min())))
