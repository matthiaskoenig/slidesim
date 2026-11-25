"""Write the Zarr v3 raster fixtures of the maps crate.

Run from the repository root:
    uvx --with 'zarr>=3.3' --with numpy python testdata/maps/make_fixtures.py
"""

import json
import shutil
from pathlib import Path

import numpy as np
import zarr
from zarr.codecs import ZstdCodec

OUT = Path(__file__).parent
ROWS, COLS, UM_PER_PX, ORIGIN = 40, 60, 2.0, [10.0, 20.0]


def linear() -> np.ndarray:
    """Value 2 + 0.5 x - 0.25 y at the pixel centres (rows = y, cols = x)."""
    r, c = np.mgrid[0:ROWS, 0:COLS]
    x = ORIGIN[0] + c * UM_PER_PX
    y = ORIGIN[1] + r * UM_PER_PX
    return (2.0 + 0.5 * x - 0.25 * y).astype(np.float32)


def write(name: str, data: np.ndarray, no_value: float | None) -> None:
    path = OUT / f"{name}.zarr"
    if path.exists():
        shutil.rmtree(path)
    arr = zarr.create_array(
        store=str(path),
        shape=data.shape,
        chunks=(16, 16),
        dtype=data.dtype,
        compressors=[ZstdCodec(level=3)],
        zarr_format=3,
    )
    arr[:] = data
    spec = {
        "id": name,
        "path": f"{name}.zarr",
        "units": "1",
        "um_per_px": UM_PER_PX,
        "origin_um": ORIGIN,
    }
    if no_value is not None:
        spec["no_value"] = no_value
    (OUT / f"{name}.json").write_text(json.dumps(spec, indent=2) + "\n")


if __name__ == "__main__":
    base = linear()
    write("linear", base, None)
    holes = base.copy()
    holes[10:13, 30:33] = np.nan
    holes[0, 0] = -1.0
    write("no_value", holes, -1.0)
