/// @file
/// Darcy velocity: u = -D grad p at the quadrature points, in 2D, with D (the permeability over
/// the viscosity, K / mu) from the point data of transport.h.
#pragma once

#include <ceed/types.h>

#include "geometry.h"
#include "transport.h"

/// u = -D grad p, with D00, D01, D11 at components 2, 3, 4 of the point data (TRANSPORT_DATA_SIZE
/// components; the others are not read).
///
/// Inputs: dp (reference gradient of the pressure, [j]), q_data (geometry.h), p_data
/// (transport.h). Output: u (the velocity per point, [c]).
CEED_QFUNCTION(Velocity2D)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*dp)[CEED_Q_VLA]     = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  const CeedScalar(*p_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  CeedScalar(*u)[CEED_Q_VLA]            = (CeedScalar(*)[CEED_Q_VLA])out[0];

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // b_jc = dX_j/dx_c
    const CeedScalar b00 = q_data[1][i], b01 = q_data[2][i], b10 = q_data[3][i], b11 = q_data[4][i];
    const CeedScalar D00 = p_data[2][i], D01 = p_data[3][i], D11 = p_data[4][i];
    // The physical gradient: dp/dx_c = sum_j dp/dX_j b_jc.
    const CeedScalar p0 = dp[0][i] * b00 + dp[1][i] * b10;
    const CeedScalar p1 = dp[0][i] * b01 + dp[1][i] * b11;

    u[0][i] = -(D00 * p0 + D01 * p1);
    u[1][i] = -(D01 * p0 + D11 * p1);
  }
  return 0;
}
