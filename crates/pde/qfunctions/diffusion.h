/// @file
/// Diffusion-reaction: the weak form of m u - div(d grad u), in 2D.
#pragma once

#include <ceed/types.h>

#include "geometry.h"

/// The coefficients: d (diffusion), m (mass: reaction rate plus the shift of an implicit time
/// step).
struct DiffusionReactionContext {
  CeedScalar diffusion;
  CeedScalar mass;
};

/// v = m w det(J) u, dv = d w det(J) J^-1 J^-T du: the operator m M + d K applied to u.
///
/// Inputs: u, du (reference gradient, [j]), q_data (geometry.h).
/// Outputs: v (against the test functions), dv (against their reference gradients, [j]).
CEED_QFUNCTION(DiffusionReaction2D)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const struct DiffusionReactionContext *c      = (const struct DiffusionReactionContext *)ctx;
  const CeedScalar                      *u      = in[0];
  const CeedScalar(*du)[CEED_Q_VLA]             = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  const CeedScalar(*q_data)[CEED_Q_VLA]         = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  CeedScalar       *v                           = out[0];
  CeedScalar(*dv)[CEED_Q_VLA]                   = (CeedScalar(*)[CEED_Q_VLA])out[1];
  const CeedScalar d = c->diffusion, m = c->mass;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar wdetJ = q_data[0][i];
    // b_jc = dX_j/dx_c
    const CeedScalar b00 = q_data[1][i], b01 = q_data[2][i], b10 = q_data[3][i], b11 = q_data[4][i];
    // The physical gradient: du/dx_c = sum_j du/dX_j b_jc.
    const CeedScalar g0 = du[0][i] * b00 + du[1][i] * b10;
    const CeedScalar g1 = du[0][i] * b01 + du[1][i] * b11;

    v[i]     = m * wdetJ * u[i];
    dv[0][i] = d * wdetJ * (b00 * g0 + b01 * g1);
    dv[1][i] = d * wdetJ * (b10 * g0 + b11 * g1);
  }
  return 0;
}
