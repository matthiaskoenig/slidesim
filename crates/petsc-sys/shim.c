/* Values PETSc defines as macros which bindgen cannot translate. Declared in wrapper.h. */
#include <petsc.h>

MPI_Comm petsc_sys_comm_self(void) { return PETSC_COMM_SELF; }
