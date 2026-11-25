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
MPI_Op petsc_sys_mpi_sum(void) { return MPI_SUM; }
MPI_Op petsc_sys_mpi_max(void) { return MPI_MAX; }

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
