# Spike: Mmg through `DMAdaptMetric`, natively and in wasm32-wasip1

**Decision: GO (option B).** All criteria G1 to G5 hold on both targets (native, wasm32-wasip1 in
wasmtime and in Node) and on both fixtures (one lobule; a square split by an interior tagged chain
into two areas, with a hole). The wasm build needs one small PETSc patch (PT-Scotch optional for
Mmg) and CMake arguments for Mmg, no source patch of Mmg. The meshes are bitwise identical on one
target across runs and, beyond G4, across all three hosts.

Date 2026-11-27, PETSc v3.26.1, Mmg 5.8.0, wasi-sdk 34.0, wasmtime 50.0.1, Node 24.21.0,
spade 2.15.1. The machine was shared and loaded during every measurement (load average 19 to 37
on 20 cores, see G5).

## Criteria

Measured with `lobule/run.sh all` (two runs per fixture and host; the numbers of both runs agree
to the last digit). Mmg parameters: PETSc's defaults (`hgrad` 1.3, `hausd` 0.01); the one retry
allowed for G3 was not needed.

| Criterion | Threshold | lobule: native | lobule: wasmtime | lobule: Node | split: native | split: wasmtime | split: Node | Verdict |
|---|---|---|---|---|---|---|---|---|
| G1 native build | `PETSC_HAVE_MMG` | yes | | | | | | pass |
| G2 wasm build, small patches | | | yes: 1 patch (12 lines) + CMake arguments | | | | | pass |
| G2 runs in wasmtime and Node | | | runs | runs | | runs | runs | pass |
| G3 sub-edges carry the segment's tag | 0 other | 0 | 0 | 0 | 0 | 0 | 0 | pass |
| G3 per-tag length vs input (max rel) | <= 1e-12 | 6.0e-16 | 6.0e-16 | 6.0e-16 | 3.8e-16 | 3.8e-16 | 3.8e-16 | pass |
| G3 per-segment length vs input (max rel) | <= 1e-12 | 3.5e-14 | 3.5e-14 | 3.5e-14 | 8.9e-15 | 8.9e-15 | 8.9e-15 | pass |
| G3 interior chain (`necrosis_border`) length rel, tag | 1e-12, kept | | | | 3.8e-16, 91/91 | 3.8e-16, 91/91 | 3.8e-16, 91/91 | pass |
| G3 cells keep their area tag | 0 other | 0 | 0 | 0 | 0 | 0 | 0 | pass |
| G3 min angle | >= 20 deg | 30.293 | 30.293 | 30.293 | 27.839 | 27.839 | 27.839 | pass |
| G3 min SICN | >= 0.4 | 0.7141 | 0.7141 | 0.7141 | 0.6975 | 0.6975 | 0.6975 | pass |
| G3 edge ratio `l / h(mid)` | in [0.4, 2.0] | 0.602 to 1.556 | same | same | 0.518 to 1.552 | same | same | pass |
| G4 hash run 1 / run 2 | identical | f99bd73767eb60cb twice | same | same | fac19a212c1efab7 twice | same | same | pass |
| G5 meshing wall time (5 runs) | native < 2 s, wasm < 10 s | 187 to 326 ms | 185 to 221 ms | 237 to 400 ms | 72 to 133 ms | 79 to 144 ms | 131 to 235 ms | pass |

Counts: lobule 4031 vertices, 7604 triangles, 11641 edges (input: 666 vertices, 874 triangles,
470 tagged sub-edges); split 2682 vertices, 5196 triangles (2114 upper, 3082 lower), 7878 edges
(input: 387 vertices, 606 triangles, 259 tagged sub-edges). No clockwise cell. The VTUs of the
three hosts are byte-identical (`cmp`).

G5 details: the meshing time is CDT + DMPlex + metric + `DMAdaptMetric` + label split, measured
inside the program; load average 19.5 at the start and the end of the timing runs. Whole
processes: native 584 to 746 ms (lobule; about 350 ms of that is OpenMPI's singleton init),
wasmtime 226 to 268 ms (precompiled `.cwasm`), Node 366 to 560 ms (including
`WebAssembly.compile` of the 16 MB module). The machine was loaded by other sessions, so the
absolute numbers are pessimistic; the margins (6x native, 25x wasm) leave no doubt. The CDT phase
of the lobule (50 to 140 ms) is dominated by the spike's brute force distance to every wall
segment for the size field, not by `spade`.

Sizes of the wasm (lobule program, release): 16,171,149 bytes raw, 15,771,053 stripped,
4,233,709 gzip -9 stripped. The same program linked against `deps/install/wasi` (PETSc without
Mmg) is 15,014,469 stripped, 3,989,267 gzip: Mmg adds 0.76 MB stripped, 0.24 MB gzip.

## The program (`lobule/`)

`lobule/src/main.rs` (raw bindgen bindings as `spike/petsc-wasm/diffusion`, `spade`, `libm` for
the circles so that the input is bitwise the same on every target):

1. Fixture geometry as tagged polylines.
   - `lobule`: hexagon of circumradius 300 um (`field_cut`), central hole radius 30 um
     (`central_wall`), six portal holes radius 25 um centered 250 um from the center towards the
     corners (`portal_wall`, holes rather than notches cut by the hexagon), circles polygonized
     at 5 um (38 and 32 segments); one area `parenchyma`.
   - `split` (controller ruling P1): square 400 um (`field_cut`), an interior kinked polyline
     from the left to the right side (`necrosis_border`, 4 segments) separating the areas
     `upper` and `lower`, a hole of radius 25 um in `lower` (`vessel_wall`).
   - `split1`: `split` with one area for both sides (to test a chain inside one area).
2. Size field (gmsh Threshold as slideviz's "5:20"): `h = 5` up to 10 um from the nearest wall
   (every tagged chain but `field_cut`, so the interior chain counts as a wall), linear to
   `h = 20` at 200 um, 20 beyond.
3. Every input segment split to the size field: `n = round(int ds / h)` sub-edges of equal
   `int ds / h`; the split points are `a + t (b - a)`.
4. `spade` CDT of all sub-edges (constraints), refined by area only (`-refine_area 700`, about an
   equilateral triangle of 40 um, twice the interior size), constraint edges kept, no angle
   limit, no outer-face exclusion (see findings). Faces are flood-filled across non-constraint
   edges into regions; a region reached from the convex hull is outside, any other one gets the
   area of the centroid of its largest face (point in polygon against the input).
5. `DMPlexCreateFromCellListPetsc`, one edge label `walls` (tag index + 1) set on the sub-edges
   via `DMPlexGetJoin`, one cell label `areas` (area index + 1).
6. `DMPlexMetricSetIsotropic`, `DMPlexMetricCreate` (indicator, P1 field), `1/h^2` per vertex,
   `DMPlexMetricCreateIsotropic`; `DMAdaptMetric(dm, metric, walls, areas)` with the options
   `-dm_adaptor mmg -dm_plex_metric_no_surf` (set by the program unless given; `no_surf` makes
   Mmg 2D mark every boundary edge and every edge between two cell references as required, so the
   walls and the interior chain are frozen).
7. Labels split back: one label per tag (value 1) from the strata of `walls`.
8. Report: counts, min angle, min SICN (`4 sqrt(3) A / sum l^2`), edge ratios against the
   analytic size field, per-tag lengths against the input polylines, per input segment: every
   output edge with both end points within 1e-9 um of the segment carries its tag and their lengths
   sum to the segment's length, every tagged edge lies on an input segment of its tag; every cell's
   area label against the input polygons; FNV-1a hash over coordinates (bits), cells, edges and
   both labels in DMPlex order; wall times; VTU (`-vtu`, tagged edges as line cells).

Options for experiments: `-refine_area`, `-refine_angle`, `-refine_exclude`, `-passes` (repeat
the adaptation with the metric recomputed on the adapted mesh), `-report_input`, `-vtu_input`,
and every `-dm_plex_metric_*` option.

## Builds

`build.sh <native|wasi>` copies `deps/src/petsc` into `deps/build/spike-mmg-<target>` and installs
into `deps/install/spike-mmg-<target>`; the prefixes the tests use are untouched. Mmg tarball:
`deps/src/mmg-v5.8.0.tar.gz` from `https://github.com/MmgTools/mmg/archive/v5.8.0.tar.gz` (the
URL of PETSc's `Mmg.py`), sha256
`686eaab84de79c072f3aedf26cd11ced44c84b435d51ce34e016ad203172922f`, 9,675,000 bytes.

### Step 1: native (G1)

```
./configure --prefix=deps/install/spike-mmg-native --with-cc=mpicc --with-cxx=0 --with-fc=0 \
  --download-f2cblaslapack=deps/src/f2cblaslapack-3.8.0.q2.tar.gz \
  --download-ptscotch=deps/src/scotch-v7.0.15.tar.gz \
  --download-ptscotch-cmake-arguments="-DTHREADS=OFF -DSCOTCH_DETERMINISTIC=FULL" \
  --download-bison=deps/src/bison-3.8.2.tar.gz \
  --download-mmg=deps/src/mmg-v5.8.0.tar.gz \
  --with-shared-libraries=1 --with-debugging=0 \
  COPTFLAGS="-O3 -DNDEBUG -march=native -ffp-contract=off -fno-fast-math"
```

Built on the first attempt (about 22 min with `-j8` under load 15 to 37), `PETSC_HAVE_MMG 1`.
Mmg's CMake found PT-Scotch through `SCOTCH_DIR` and compiled with `-DUSE_SCOTCH`. CMake picked the
host `c++` for Mmg's `inoutcpp_*.cpp` (VTK I/O stubs, always compiled) although PETSc has
`--with-cxx=0`; harmless natively.

### Step 2: wasm32-wasip1 (G2), every attempt

The base is the configure of `spike/petsc-wasm/build-wasi.sh` with its five PETSc patches
(`spike/petsc-wasm/patches/petsc-*.patch`).

1. **PT-Scotch on MPIUNI** (`WASI_PTSCOTCH=1 build.sh wasi`: `--download-ptscotch`, its CMake
   arguments, `--download-bison --download-bison-cc=gcc` for a host bison): PETSc's configure
   accepts PT-Scotch with MPIUNI (`PTSCOTCH.py` has no `noMPIUni`), but Scotch's CMake fails:
   `Could NOT find MPI_C (missing: MPI_C_WORKS)` (log `logs/build-wasi-1-ptscotch.log`, not
   committed). A standalone probe of Scotch 7.0.15 with wasi-sdk then showed what else this path
   needs: FindMPI overrides (`MPI_C_WORKS`, `MPI_C_HEADER_DIR` on MPIUNI's `mpi.h`,
   `MPI_C_VERSION`), `<sys/wait.h>` (absent in wasi-libc; `-DHAVE_NOT_SYS_WAIT_H`), running
   `dummysizes`/`ptdummysizes` at build time (wasm executables; the custom command uses
   `$<TARGET_FILE:...>`, so `CMAKE_CROSSCOMPILING_EMULATOR` is not applied: a Scotch CMake patch),
   four MPI functions PT-Scotch calls and MPIUNI lacks (`MPI_Get_address`, `MPI_Ialltoallv`,
   `MPI_Type_create_hindexed`, `MPI_Type_struct`: a PETSc patch adding them), and the MPIUNI
   symbols live in `libpetsc` (link cycle). Not small: abandoned.
2. **Mmg without PT-Scotch** (patch `patches/petsc-wasi-0001-mmg-ptscotch-optional.patch`:
   `Mmg.py` takes PT-Scotch as an optional dependency and passes `-DUSE_SCOTCH=OFF` without it):
   PETSc configured Mmg, Mmg's CMake failed: `M_LIB` NOTFOUND (`find_library(M_LIB m)` searches
   the host paths, not the sysroot).
3. A standalone probe of Mmg 5.8.0 with wasi-sdk found the rest in two rounds (about 30 s each):
   - `inoutcpp_*.cpp` compiled by `clang++` of wasi-sdk without the target flags and the WASI
     emulation defines: `signal.h`/`sys/resource.h` errors. Fix: `CMAKE_CXX_COMPILER` and
     `CMAKE_CXX_FLAGS`.
   - `error: common symbols are not yet implemented for Wasm: ddb`: Mmg adds `-fcommon` to its
     libraries for clang > 10 (`cmake/modules/macros.cmake`), and `int8_t ddb;` in
     `mmg3d1_delone.c` is a tentative definition. Fix: `-fno-common` after Mmg's flags by the
     compile rule `CMAKE_C_COMPILE_OBJECT` (natively gcc already compiles Mmg with -fno-common,
     its default since gcc 10, so this is proven safe).
   - Mmg's executables (`mmg2d_O3`, `mmgs_O3`, `mmg3d_O3`, `genheader`) fail to link: `system()`
     does not exist in wasi-libc, `signal()` needs the emulation library. They are built but
     never used: `CMAKE_EXE_LINKER_FLAGS=-Wl,--allow-undefined`. `genheader` would also be run
     (wasm) to write the Fortran header if Perl is found: `CMAKE_DISABLE_FIND_PACKAGE_Perl=ON`.
   - The libraries then build; `libmmg.a` needs only `signal` and `getrusage` from outside,
     both covered by the emulation libraries PETSc already links.
4. Full build with all of the above: interrupted by `No space left on device` (the shared disk
   ran full from other sessions; no change).
5. Same build again: **success** (about 8 min), `PETSC_HAVE_MMG 1`, static `libmmg*.a` in the
   prefix.

Final wasm configure (in `build.sh`):

```
./configure PETSC_ARCH=arch-wasi --prefix=deps/install/spike-mmg-wasi \
  --with-cc=$WASI_SDK_PATH/bin/clang --with-cxx=0 --with-fc=0 --with-mpi=0 \
  --with-ar=$WASI_SDK_PATH/bin/llvm-ar --with-ranlib=$WASI_SDK_PATH/bin/llvm-ranlib \
  CFLAGS+="--target=wasm32-wasip1 --sysroot=$WASI_SDK_PATH/share/wasi-sysroot" \
  CPPFLAGS="-D_WASI_EMULATED_SIGNAL -D_WASI_EMULATED_PROCESS_CLOCKS -D_WASI_EMULATED_GETPID" \
  COPTFLAGS="-O3 -DNDEBUG -ffp-contract=off -fno-fast-math -msimd128" \
  LIBS="-lwasi-emulated-signal -lwasi-emulated-process-clocks -lwasi-emulated-getpid" \
  --download-f2cblaslapack=deps/src/f2cblaslapack-3.8.0.q2.tar.gz \
  --download-mmg=deps/src/mmg-v5.8.0.tar.gz \
  --download-mmg-cmake-arguments="-DCMAKE_DISABLE_FIND_PACKAGE_Perl=ON \
    -DM_LIB=$WASI_SDK_PATH/share/wasi-sysroot/lib/wasm32-wasip1/libm.a \
    -DCMAKE_CXX_COMPILER=$WASI_SDK_PATH/bin/clang++ \
    -DCMAKE_CXX_FLAGS=\"<CFLAGS+ target> <COPTFLAGS> <CPPFLAGS>\" \
    -DCMAKE_C_COMPILE_OBJECT='<CMAKE_C_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -fno-common -o <OBJECT> -c <SOURCE>' \
    -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-undefined" \
  --with-shared-libraries=0 --with-x=0 --with-ssl=0 --with-pthread=0 \
  --with-debugging=0 --with-avx512-kernels=0 --with-batch \
  --known-sdot-returns-double=0 --known-snrm2-returns-double=0
```

plus the patches `spike/petsc-wasm/patches/petsc-000{1..5}-*.patch` and
`patches/petsc-wasi-0001-mmg-ptscotch-optional.patch`.

## Patches

| Patch | Reason |
|---|---|
| `patches/petsc-wasi-0001-mmg-ptscotch-optional.patch` | `Mmg.py` requires PT-Scotch, whose package requires an MPI that Scotch's CMake can use (not MPIUNI, see attempt 1). Mmg uses Scotch only for renumbering in mmgs and mmg3d (`USE_SCOTCH`), never in mmg2d. The patch moves PT-Scotch to `odeps` and passes `-DUSE_SCOTCH=OFF` when it is absent; with PT-Scotch present (native) nothing changes. Upstreamable. |

No source patch of Mmg.

## Findings

- **2D Mmg does not use Scotch.** `grep USE_SCOTCH src/mmg2d` is empty; native (Mmg with
  Scotch) and wasm (without) run the same 2D code and give bitwise identical meshes.
- **Cross-target bitwise identity.** The meshes (hash and VTU bytes) are identical natively, in
  wasmtime and in Node for both fixtures. Not a criterion, but it means the browser and the
  native run can start from the same mesh.
- **Frozen walls and interior chains.** With `-dm_plex_metric_no_surf`, mmg2d marks boundary edges
  and edges between triangles of different references as required (`analys_2d.c`), so the
  interior chain survives because the two areas carry different cell labels. Between two cells
  of the *same* reference mmg2d strips the edge's `MG_REF`/`MG_BDY` tags: the fixture `split1`
  (the split fixture with both areas merged into one) loses the whole `necrosis_border`
  (0 edges, length rel 1.0; everything else passes). PETSc's adaptor exposes neither Mmg's
  required edges nor `opnbdy`. Workaround for the mesher: the cell label given to Mmg is the
  region id of the flood fill (every region bounded by tagged chains gets its own value), mapped
  back to the area tags afterwards; then every tagged chain that closes or runs from wall to wall
  separates two references and is kept. A dangling chain ending inside an area would still be
  lost (a crack has the same region on both sides).
- **Metric interpolation needs a fine enough background.** Mmg interpolates the vertex metric
  linearly on the input mesh for every vertex it inserts. With a coarse `spade` mesh the
  interpolated metric undershoots the analytic size field between a wall vertex (h = 5) and a far
  vertex: in the first configuration (`spade` refinement with a 20 deg angle limit and
  `exclude_outer_faces(true)`) the split fixture got no refinement at all and three interior
  edges with ratio 0.361 to 0.392 (G3 fail, all else passing). Refining the CDT by area fixes it
  (a second adaptation pass with the metric recomputed on the adapted mesh fixes it too, at
  twice the adaptation time).
- **`spade`'s `exclude_outer_faces` does not support interior chains.** It classifies faces by
  the parity of crossed constraint edges (layer peeling), which is wrong for a chain separating
  two areas: on the split fixture it excluded everything from refinement. The program therefore
  refines without exclusion and classifies the faces itself (flood fill across non-constraint
  edges). Refining without exclusion also refines inside the holes (wasted, dropped).
- **Convex hull slivers.** The split points of a straight segment are not exactly collinear in
  floating point; the CDT then has flat faces between the sub-edges and the convex hull whose
  centroids lie on the boundary. Classifying faces by their own centroid took these in (the
  first run: min angle 0, `field_cut` length off by 43 %); the region flood fill avoids it.
- **Mmg in wasm prints to stdout.** `MMG5_memSize` knows no WASI (`__wasi__` is neither unix nor
  Windows), warns on stderr ("unknown system") and `MMG5_memOption_memSet` prints
  "Maximum memory set to default value: 800 MB." to **stdout**, twice per adaptation, regardless
  of the verbosity. Natively silent. See Task 2.
- `spade` refinement iterates no hash set (only its returned `excluded_faces` list is in hash
  order, unused here), so the CDT is deterministic.

## What Task 2 needs

- versions.env: `MMG_VERSION=5.8.0`, `MMG_SHA256=686eaab84de79c072f3aedf26cd11ced44c84b435d51ce34e016ad203172922f`,
  tarball `https://github.com/MmgTools/mmg/archive/v5.8.0.tar.gz` saved as
  `deps/src/mmg-v5.8.0.tar.gz`; fetch.sh checks `Mmg.py`'s `self.version = '5.8.0'`.
- Native: the configure of Step 1 (`--download-mmg=<tarball>`); recommended additionally the
  patch above in every build and `--download-mmg-cmake-arguments=-DUSE_SCOTCH=OFF`, so that
  Mmg is configured the same on every target (no difference in 2D today, but mmg3d renumbers
  with Scotch and milestone 7 meshes in 3D).
- wasm: the configure and patch above.
- The stdout print of Mmg in wasm: either a PETSc patch of `mmgadapt.c` that sets
  `MMG2D_IPARAM_mem`/`MMG3D_IPARAM_mem` under `__wasm__` (then `memOption_memSet` takes the
  user branch and stays silent on stdout; the stderr warning of `MMG5_memSize` stays), or build
  Mmg outside PETSc's download (like libCEED, then `--with-mmg-dir`) with a two-line Mmg patch
  (`MMG5_memSize` for `__wasi__`) that would also replace the `CMAKE_C_COMPILE_OBJECT` override
  by dropping `-fcommon`. The second is cleaner.
- The mesher (Task 4): refine the `spade` CDT by area (about twice the interior size) without
  `exclude_outer_faces`, classify faces by a flood fill over non-constraint edges, freeze the
  boundary with `-dm_plex_metric_no_surf`, give Mmg the region id (not the area tag) as cell
  reference so that every tagged interior chain separates two references.

## Files

- `build.sh`: the two PETSc builds.
- `patches/petsc-wasi-0001-mmg-ptscotch-optional.patch`.
- `lobule/`: the program (`run.sh [native|wasi|all] [options]` builds, runs both fixtures twice
  per host, compares hashes, prints the wasm sizes; outputs and VTUs in `lobule/target/runs`).
- `run-node.mjs`: Node WASI runner with `/out` preopened for the VTU.
