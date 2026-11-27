/* Values PETSc defines as macros which bindgen cannot translate, and calls into PETSc's external
   packages. Declared in wrapper.h. */
#include <petsc.h>
#if defined(PETSC_HAVE_PTSCOTCH)
  #include <scotch.h>
#endif

MPI_Comm petsc_sys_comm_self(void) { return PETSC_COMM_SELF; }
MPI_Comm petsc_sys_comm_world(void) { return MPI_COMM_WORLD; }

/* Data types and reduction operations of MPI_Allreduce (Open MPI defines them as addresses of
   its globals, MPICH as casted integers). */
MPI_Datatype petsc_sys_mpiu_int(void) { return MPIU_INT; }
MPI_Datatype petsc_sys_mpiu_real(void) { return MPIU_REAL; }
/* The data type of MPI_Bcast of bytes. */
MPI_Datatype petsc_sys_mpi_byte(void) { return MPI_BYTE; }
MPI_Op petsc_sys_mpi_sum(void) { return MPI_SUM; }
MPI_Op petsc_sys_mpi_max(void) { return MPI_MAX; }
MPI_Op petsc_sys_mpi_replace(void) { return MPI_REPLACE; }

/* MPI_Gather of one int per process and MPI_Gatherv of bytes to `root`. Functions here, not
   bindings of MPI's own: MPIUNI (PETSc without MPI, as in the wasm build) defines the MPI
   collectives as macros, which bindgen cannot translate. */
int petsc_sys_mpi_gather_int(int value, int *values, int root, MPI_Comm comm)
{
  return MPI_Gather(&value, 1, MPI_INT, values, 1, MPI_INT, root, comm);
}
int petsc_sys_mpi_gatherv_bytes(const void *data, int count, void *gathered, const int *counts, const int *offsets, int root, MPI_Comm comm)
{
  return MPI_Gatherv(data, count, MPI_BYTE, gathered, counts, offsets, MPI_BYTE, root, comm);
}

/* PetscFree of an array PETSc allocated for the caller (PetscFree is a macro). */
PetscErrorCode petsc_sys_free(void *p) { return PetscFree(p); }

/* Resets the random generator of PT-Scotch to its initial (fixed) seed (SCOTCH_randomReset), if
   PETSc has PT-Scotch. Scotch's generator is global and continues from one partitioning to the
   next: without a reset, partitioning the same graph again can give another partition. */
void petsc_sys_ptscotch_random_reset(void)
{
#if defined(PETSC_HAVE_PTSCOTCH)
  SCOTCH_randomReset();
#endif
}
