/* Values PETSc defines as macros which bindgen cannot translate, and calls into PETSc's external
   packages. Declared in wrapper.h. */
#include <petsc.h>
#include <string.h>
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

/* The MPI calls of the Rust code, as functions: MPIUNI (PETSc without MPI, as in the wasm build:
   one process) defines them as macros, which bindgen cannot translate. */
int petsc_sys_mpi_comm_rank(MPI_Comm comm, int *rank) { return MPI_Comm_rank(comm, rank); }
int petsc_sys_mpi_comm_size(MPI_Comm comm, int *size) { return MPI_Comm_size(comm, size); }
int petsc_sys_mpi_allreduce(const void *send, void *receive, int count, MPI_Datatype datatype,
                            MPI_Op op, MPI_Comm comm)
{
  return MPI_Allreduce(send, receive, count, datatype, op, comm);
}
int petsc_sys_mpi_bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
{
  return MPI_Bcast(buffer, count, datatype, root, comm);
}
int petsc_sys_mpi_barrier(MPI_Comm comm) { return MPI_Barrier(comm); }
int petsc_sys_mpi_abort(MPI_Comm comm, int code) { return MPI_Abort(comm, code); }

/* MPI_Gather of one int per process and MPI_Gatherv of bytes to `root`, as the calls above. */
int petsc_sys_mpi_gather_int(int value, int *values, int root, MPI_Comm comm)
{
  return MPI_Gather(&value, 1, MPI_INT, values, 1, MPI_INT, root, comm);
}
int petsc_sys_mpi_gatherv_bytes(const void *data, int count, void *gathered, const int *counts,
                                const int *offsets, int root, MPI_Comm comm)
{
  return MPI_Gatherv(data, count, MPI_BYTE, gathered, counts, offsets, MPI_BYTE, root, comm);
}

/* MPI_Alltoall of one int per process pair and MPI_Alltoallv of bytes, as the Gather wrappers
   above. MPIUNI's MPI_Alltoallv aborts: with one process it is the copy of the bytes to the
   process itself. */
int petsc_sys_mpi_alltoall_int(const int *sends, int *receives, MPI_Comm comm)
{
  return MPI_Alltoall(sends, 1, MPI_INT, receives, 1, MPI_INT, comm);
}
int petsc_sys_mpi_alltoallv_bytes(const void *sends, const int *send_counts,
                                  const int *send_offsets, void *receives,
                                  const int *receive_counts, const int *receive_offsets,
                                  MPI_Comm comm)
{
#if defined(PETSC_HAVE_MPIUNI)
  (void)comm;
  (void)receive_counts;
  if (send_counts[0] > 0)
    memcpy((char *)receives + receive_offsets[0], (const char *)sends + send_offsets[0],
           (size_t)send_counts[0]);
  return MPI_SUCCESS;
#else
  return MPI_Alltoallv(sends, send_counts, send_offsets, MPI_BYTE, receives, receive_counts,
                       receive_offsets, MPI_BYTE, comm);
#endif
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
