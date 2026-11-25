/// @file
/// Mass terms: integrals of a field times the test functions; and the values of a field at the
/// points.
#pragma once

#include <ceed/types.h>

#include "geometry.h"

/// v = w det(J) u: with u interpolated to the points, the mass matrix applied to u; with u given
/// at the points (a source), its integral against the test functions.
///
/// Inputs: q_data (geometry.h), u. Output: v (against the test functions).
CEED_QFUNCTION(Mass)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar *u                   = in[1];
  CeedScalar       *v                   = out[0];

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) { v[i] = q_data[0][i] * u[i]; }
  return 0;
}

/// The field at the points: with u interpolated to the points, its values there.
///
/// Input: u. Output: u_q (per point).
CEED_QFUNCTION(Values)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *u   = in[0];
  CeedScalar       *u_q = out[0];

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) { u_q[i] = u[i]; }
  return 0;
}
