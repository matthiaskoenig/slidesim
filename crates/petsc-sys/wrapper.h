/* Headers for bindgen: the whole PETSc API, plus the shim (shim.c). */
#include <petsc.h>

/* PETSC_COMM_SELF (MPI_COMM_SELF is a macro whose value depends on the MPI implementation). */
MPI_Comm petsc_sys_comm_self(void);
