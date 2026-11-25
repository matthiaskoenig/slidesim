/* Headers for bindgen: the whole PETSc API, plus the shim (shim.c). */
#include <petsc.h>

/* PETSC_COMM_SELF (MPI_COMM_SELF is a macro whose value depends on the MPI implementation). */
MPI_Comm petsc_sys_comm_self(void);
/* MPI_COMM_WORLD (a macro too), for MPI_Abort. */
MPI_Comm petsc_sys_comm_world(void);

/* MPI_Allreduce arguments: MPIU_INT, MPIU_REAL, MPI_SUM, MPI_MAX. */
MPI_Datatype petsc_sys_mpiu_int(void);
MPI_Datatype petsc_sys_mpiu_real(void);
MPI_Op petsc_sys_mpi_sum(void);
MPI_Op petsc_sys_mpi_max(void);

/* PetscFree of an array PETSc allocated for the caller. */
PetscErrorCode petsc_sys_free(void *p);

/* SCOTCH_randomReset if PETSc has PT-Scotch, else nothing. */
void petsc_sys_ptscotch_random_reset(void);
