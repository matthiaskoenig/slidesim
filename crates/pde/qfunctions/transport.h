/// @file
/// Transport: the weak form of shift m u + r u + vel . grad u - div(D grad u), in 2D, with the
/// coefficients given at the quadrature points, stabilized by SUPG (streamline upwind
/// Petrov-Galerkin) where the velocity vel is not zero; and the integral of a source against the
/// same test functions.
#pragma once

#include <ceed/types.h>

#include "geometry.h"

/// The components of the point data per point: m (the factor of u_t, for example the porosity),
/// r (the reaction, for example porosity times decay rate), the symmetric diffusion tensor D as
/// D00, D01, D11, the velocity vel0, vel1 and the SUPG parameter tau (s; zero where the velocity
/// is zero, and zero without stabilization).
#define TRANSPORT_DATA_SIZE 8

/// The terms the operator applies (TransportContext.terms): the steady terms
/// r u + vel . grad u - div(D grad u); the mass term m u; or the Jacobian of an implicit step,
/// shift m u plus the steady terms.
#define TRANSPORT_STEADY 0
#define TRANSPORT_MASS 1
#define TRANSPORT_JACOBIAN 2

/// The context: the shift of an implicit time step (for TRANSPORT_JACOBIAN), and the terms
/// (TRANSPORT_STEADY, TRANSPORT_MASS or TRANSPORT_JACOBIAN, as a scalar).
struct TransportContext {
  CeedScalar shift;
  CeedScalar terms;
};

/// The weak form against the SUPG test function v + tau vel . grad v, with the residual of the
/// terms without diffusion (P1: the second derivatives vanish in each cell):
///
/// int (a m + b r) u v + b (vel . grad u) v + b (D grad u) . grad v
///   + tau (vel . grad v) ((a m + b r) u + b vel . grad u),
///
/// with a = 0, b = 1 for the steady terms; a = 1, b = 0 for the mass term
/// (int m u (v + tau vel . grad v)); and a = shift, b = 1 for the Jacobian.
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
    const CeedScalar vel0 = p_data[5][i], vel1 = p_data[6][i], tau = p_data[7][i];
    // The physical gradient: du/dx_c = sum_j du/dX_j b_jc.
    const CeedScalar g0 = du[0][i] * b00 + du[1][i] * b10;
    const CeedScalar g1 = du[0][i] * b01 + du[1][i] * b11;
    // The residual of the terms without diffusion, which the test function v takes, and SUPG's
    // tau vel . grad v.
    const CeedScalar residual = (a * m + b * r) * u[i] + b * (vel0 * g0 + vel1 * g1);
    const CeedScalar supg     = tau * residual;
    // The flux against grad v: the diffusive flux D grad u of the steady terms, and SUPG.
    const CeedScalar f0 = b * (D00 * g0 + D01 * g1) + supg * vel0;
    const CeedScalar f1 = b * (D01 * g0 + D11 * g1) + supg * vel1;

    v[i]     = wdetJ * residual;
    dv[0][i] = wdetJ * (b00 * f0 + b01 * f1);
    dv[1][i] = wdetJ * (b10 * f0 + b11 * f1);
  }
  return 0;
}

/// The integral of a source f given at the points against the SUPG test function of
/// Transport2D: v = w det(J) f, dv = w det(J) tau f J^-1 vel, that is int f (v + tau vel . grad v).
///
/// Inputs: q_data (geometry.h), p_data (TRANSPORT_DATA_SIZE; the velocity and tau), f. Outputs:
/// v (against the test functions), dv (against their reference gradients, [j]).
CEED_QFUNCTION(TransportSource2D)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*p_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  const CeedScalar *f                   = in[2];
  CeedScalar       *v                   = out[0];
  CeedScalar(*dv)[CEED_Q_VLA]           = (CeedScalar(*)[CEED_Q_VLA])out[1];

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar wdetJ = q_data[0][i];
    // b_jc = dX_j/dx_c
    const CeedScalar b00 = q_data[1][i], b01 = q_data[2][i], b10 = q_data[3][i], b11 = q_data[4][i];
    const CeedScalar vel0 = p_data[5][i], vel1 = p_data[6][i], tau = p_data[7][i];
    const CeedScalar supg = wdetJ * tau * f[i];

    v[i]     = wdetJ * f[i];
    dv[0][i] = supg * (b00 * vel0 + b01 * vel1);
    dv[1][i] = supg * (b10 * vel0 + b11 * vel1);
  }
  return 0;
}
