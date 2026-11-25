/// @file
/// Transport: the weak form of shift m u + r u - div(D grad u), in 2D, with the coefficients
/// given at the quadrature points.
#pragma once

#include <ceed/types.h>

#include "geometry.h"

/// The components of the point data per point: m (the factor of u_t, for example the porosity),
/// r (the reaction, for example porosity times decay rate), the symmetric diffusion tensor D as
/// D00, D01, D11, the velocity u0, u1 and the SUPG parameter tau (velocity and tau for
/// advection, which is not applied yet).
#define TRANSPORT_DATA_SIZE 8

/// The terms the operator applies (TransportContext.terms): the steady terms r u - div(D grad u);
/// the mass term m u; or the Jacobian of an implicit step, shift m u plus the steady terms.
#define TRANSPORT_STEADY 0
#define TRANSPORT_MASS 1
#define TRANSPORT_JACOBIAN 2

/// The context: the shift of an implicit time step (for TRANSPORT_JACOBIAN), and the terms
/// (TRANSPORT_STEADY, TRANSPORT_MASS or TRANSPORT_JACOBIAN, as a scalar).
struct TransportContext {
  CeedScalar shift;
  CeedScalar terms;
};

/// v = (a m + b r) w det(J) u, dv = b w det(J) J^-1 D J^-T du, with a = 0, b = 1 for the steady
/// terms; a = 1, b = 0 for the mass term; and a = shift, b = 1 for the Jacobian.
///
/// Inputs: u, du (reference gradient, [j]), q_data (geometry.h), p_data (TRANSPORT_DATA_SIZE).
/// Outputs: v (against the test functions), dv (against their reference gradients, [j]).
CEED_QFUNCTION(Transport2D)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const struct TransportContext *c      = (const struct TransportContext *)ctx;
  const CeedScalar              *u      = in[0];
  const CeedScalar(*du)[CEED_Q_VLA]     = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  const CeedScalar(*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  const CeedScalar(*p_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[3];
  CeedScalar       *v                   = out[0];
  CeedScalar(*dv)[CEED_Q_VLA]           = (CeedScalar(*)[CEED_Q_VLA])out[1];
  const CeedScalar a = c->terms == TRANSPORT_MASS ? 1.0 : (c->terms == TRANSPORT_JACOBIAN ? c->shift : 0.0);
  const CeedScalar b = c->terms == TRANSPORT_MASS ? 0.0 : 1.0;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar wdetJ = q_data[0][i];
    // b_jc = dX_j/dx_c
    const CeedScalar b00 = q_data[1][i], b01 = q_data[2][i], b10 = q_data[3][i], b11 = q_data[4][i];
    const CeedScalar m = p_data[0][i], r = p_data[1][i];
    const CeedScalar D00 = p_data[2][i], D01 = p_data[3][i], D11 = p_data[4][i];
    // The physical gradient: du/dx_c = sum_j du/dX_j b_jc.
    const CeedScalar g0 = du[0][i] * b00 + du[1][i] * b10;
    const CeedScalar g1 = du[0][i] * b01 + du[1][i] * b11;
    // The diffusive flux D grad u, of the steady terms.
    const CeedScalar f0 = b * (D00 * g0 + D01 * g1);
    const CeedScalar f1 = b * (D01 * g0 + D11 * g1);

    v[i]     = (a * m + b * r) * wdetJ * u[i];
    dv[0][i] = wdetJ * (b00 * f0 + b01 * f1);
    dv[1][i] = wdetJ * (b10 * f0 + b11 * f1);
  }
  return 0;
}
