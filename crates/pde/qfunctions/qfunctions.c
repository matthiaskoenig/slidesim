/* The QFunctions of the headers here, exported for Rust: a CEED_QFUNCTION is static. Each
   wrapper also refers to the location string CEED_QFUNCTION defines, which is otherwise unused. */
#include "geometry.h"
#include "mass.h"
#include "transport.h"
#include "velocity.h"

#define EXPORT(name)                                                                                      \
  int slidesim_pde_##name(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) { \
    (void)name##_loc;                                                                                     \
    return name(ctx, Q, in, out);                                                                         \
  }

EXPORT(SetupGeometry2D)
EXPORT(QuadraturePoints2D)
EXPORT(Mass)
EXPORT(Values)
EXPORT(Transport2D)
EXPORT(Velocity2D)
