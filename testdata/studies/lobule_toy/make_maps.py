"""Write the zonation map of the study lobule_toy: Zarr v3, zstd, float32 (rows = y, cols = x).

Run from the repository root:
    uvx --with 'zarr>=3.3' --with numpy python testdata/studies/lobule_toy/make_maps.py

The value `p` is linear along the line from the centre of the portal hole to the centre of the
central hole of `testdata/meshes/lobule_toy.msh`: 0 at the portal one, 1 at the central one, the
projection of the point on that line (clipped to [0, 1]). The raster covers the square
[0, 400] x [0, 400] um with a margin of 8 um.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import zarr
from zarr.codecs import ZstdCodec

OUT = Path(__file__).resolve().parent
PORTAL = np.array([90.0, 90.0])
CENTRAL = np.array([310.0, 310.0])
UM_PER_PX = 4.0
ORIGIN = [-8.0, -8.0]
SIZE = 106


def zonation() -> np.ndarray:
    r, c = np.mgrid[0:SIZE, 0:SIZE]
    x = ORIGIN[0] + c * UM_PER_PX
    y = ORIGIN[1] + r * UM_PER_PX
    axis = CENTRAL - PORTAL
    p = ((x - PORTAL[0]) * axis[0] + (y - PORTAL[1]) * axis[1]) / axis.dot(axis)
    return np.clip(p, 0.0, 1.0).astype(np.float32)


if __name__ == "__main__":
    path = OUT / "zonation.zarr"
    if path.exists():
        shutil.rmtree(path)
    arr = zarr.create_array(
        store=str(path),
        shape=(SIZE, SIZE),
        chunks=(32, 32),
        dtype="float32",
        compressors=[ZstdCodec(level=3)],
        zarr_format=3,
    )
    arr[:] = zonation()
    print(json.dumps({"id": "zonation", "path": "zonation.zarr", "um_per_px": UM_PER_PX, "origin_um": ORIGIN}))
