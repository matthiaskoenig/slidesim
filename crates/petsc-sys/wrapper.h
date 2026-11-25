/* Headers for bindgen: the whole PETSc API, plus the shim (shim.c). */
#include <petsc.h>

/* PETSC_COMM_SELF (MPI_COMM_SELF is a macro whose value depends on the MPI implementation). */
MPI_Comm petsc_sys_comm_self(void);

/* MPI_Allreduce arguments: MPIU_INT, MPIU_REAL, MPI_SUM, MPI_MAX. */
MPI_Datatype petsc_sys_mpiu_int(void);
MPI_Datatype petsc_sys_mpiu_real(void);
MPI_Op petsc_sys_mpi_sum(void);
MPI_Op petsc_sys_mpi_max(void);
