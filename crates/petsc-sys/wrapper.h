/* Headers for bindgen: the whole PETSc API, plus the shim (shim.c). */
#include <petsc.h>

/* PETSC_COMM_SELF (MPI_COMM_SELF is a macro whose value depends on the MPI implementation). */
MPI_Comm petsc_sys_comm_self(void);
/* MPI_COMM_WORLD (a macro too), for MPI_Abort. */
MPI_Comm petsc_sys_comm_world(void);

/* MPI_Allreduce arguments: MPIU_INT, MPIU_REAL, MPI_SUM, MPI_MAX. */
MPI_Datatype petsc_sys_mpiu_int(void);
MPI_Datatype petsc_sys_mpiu_real(void);
/* MPI_BYTE, for MPI_Bcast of bytes. */
MPI_Datatype petsc_sys_mpi_byte(void);
MPI_Op petsc_sys_mpi_sum(void);
MPI_Op petsc_sys_mpi_max(void);
/* MPI_REPLACE, for PetscSFBcast. */
MPI_Op petsc_sys_mpi_replace(void);

/* MPI_Gather of one int per process, MPI_Gatherv of bytes (MPIUNI defines both as macros). */
int petsc_sys_mpi_gather_int(int value, int *values, int root, MPI_Comm comm);
int petsc_sys_mpi_gatherv_bytes(const void *data, int count, void *gathered, const int *counts, const int *offsets, int root, MPI_Comm comm);

/* MPI_Alltoall of one int per process pair, MPI_Alltoallv of bytes (macros in MPIUNI too). */
int petsc_sys_mpi_alltoall_int(const int *sends, int *receives, MPI_Comm comm);
int petsc_sys_mpi_alltoallv_bytes(const void *sends, const int *send_counts, const int *send_offsets, void *receives, const int *receive_counts, const int *receive_offsets, MPI_Comm comm);

/* PetscFree of an array PETSc allocated for the caller. */
PetscErrorCode petsc_sys_free(void *p);

/* SCOTCH_randomReset if PETSc has PT-Scotch, else nothing. */
void petsc_sys_ptscotch_random_reset(void);
