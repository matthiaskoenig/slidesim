/* Values PETSc defines as macros which bindgen cannot translate. Declared in wrapper.h. */
#include <petsc.h>

MPI_Comm petsc_sys_comm_self(void) { return PETSC_COMM_SELF; }

/* Data types and reduction operations of MPI_Allreduce (Open MPI defines them as addresses of
   its globals, MPICH as casted integers). */
MPI_Datatype petsc_sys_mpiu_int(void) { return MPIU_INT; }
MPI_Datatype petsc_sys_mpiu_real(void) { return MPIU_REAL; }
MPI_Op petsc_sys_mpi_sum(void) { return MPI_SUM; }
MPI_Op petsc_sys_mpi_max(void) { return MPI_MAX; }

/* PetscFree of an array PETSc allocated for the caller (PetscFree is a macro). */
PetscErrorCode petsc_sys_free(void *p) { return PetscFree(p); }
