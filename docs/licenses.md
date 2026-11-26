# Licenses

slidesim is MIT licensed ([`LICENSE`](https://github.com/matthiaskoenig/slidesim/blob/main/LICENSE)).

## Third-party licenses

The native dependencies are linked into slidesim (`deps/versions.env` pins their versions):

| Dependency | License |
| --- | --- |
| PETSc | BSD-2-Clause |
| libCEED | BSD-2-Clause |
| PT-Scotch (Scotch) | CeCILL-C |
| f2cblaslapack (PETSc's C translation of the reference BLAS and LAPACK) | LAPACK's modified BSD license |

bison and flex are build tools of PT-Scotch only and are not linked. MPI, and the compression
libraries Scotch uses, come from the system.

The cell models run on [sbmlsim-core](https://github.com/matthiaskoenig/sbmlsim-core) (MIT),
whose engine links SUNDIALS (BSD-3-Clause) and parts of SuiteSparse (KLU and BTF:
LGPL-2.1-or-later; AMD and COLAMD: BSD-3-Clause); its
[licenses page](https://matthiaskoenig.github.io/sbmlsim-core/licenses/) has the details.

The 3D octree meshes planned for a later milestone need p4est (GPL-2.0-or-later); it will be
linked only in an optional `octree` feature build, whose binaries are GPL. The default builds
stay MIT.
