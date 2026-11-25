/// @file
/// Geometry of P1 triangles at quadrature points (2D), for the operators of mass.h and
/// transport.h.
#pragma once

#include <ceed/types.h>

/// The components of the quadrature data per point: w det(J), then the inverse Jacobian
/// dX_j/dx_c (reference coordinates X, physical coordinates x) at 1 + 2 j + c.
#define GEOMETRY_2D_SIZE 5

/// The quadrature data of the points (see GEOMETRY_2D_SIZE).
///
/// Inputs: dx (the Jacobian of the coordinates, dx_c/dX_j at [j][c]), w (quadrature weights).
/// Output: q_data.
///
/// A non-positive det(J) (an inverted or degenerate element) is stored as it is, for the caller
/// to find.
CEED_QFUNCTION(SetupGeometry2D)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*J)[2][CEED_Q_VLA] = (const CeedScalar(*)[2][CEED_Q_VLA])in[0];
  const CeedScalar *w                 = in[1];
  CeedScalar(*q_data)[CEED_Q_VLA]     = (CeedScalar(*)[CEED_Q_VLA])out[0];

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // a_cj = dx_c/dX_j
    const CeedScalar a00 = J[0][0][i], a01 = J[1][0][i], a10 = J[0][1][i], a11 = J[1][1][i];
    const CeedScalar det = a00 * a11 - a01 * a10;

    q_data[0][i] = w[i] * det;
    q_data[1][i] = a11 / det;
    q_data[2][i] = -a01 / det;
    q_data[3][i] = -a10 / det;
    q_data[4][i] = a00 / det;
  }
  return 0;
}

/// The physical coordinates of the quadrature points.
///
/// Input: x (the coordinates interpolated to the points, [c]). Output: the same, stored per point.
CEED_QFUNCTION(QuadraturePoints2D)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*x)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  CeedScalar(*x_q)[CEED_Q_VLA]     = (CeedScalar(*)[CEED_Q_VLA])out[0];

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    x_q[0][i] = x[0][i];
    x_q[1][i] = x[1][i];
  }
  return 0;
}
