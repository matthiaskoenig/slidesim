//! `CeedBasis`: H1 bases on reference elements, with their quadrature.

use std::ptr::{null, null_mut};
use std::slice;

use crate::{Ceed, Error, Result, chk, sys, to_int};

/// A reference element shape (`CeedElemTopology`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Topology {
    Line,
    Triangle,
    Quadrilateral,
    Tetrahedron,
    Pyramid,
    Prism,
    Hexahedron,
}

impl Topology {
    fn raw(self) -> sys::CeedElemTopology {
        match self {
            Topology::Line => sys::CEED_TOPOLOGY_LINE,
            Topology::Triangle => sys::CEED_TOPOLOGY_TRIANGLE,
            Topology::Quadrilateral => sys::CEED_TOPOLOGY_QUAD,
            Topology::Tetrahedron => sys::CEED_TOPOLOGY_TET,
            Topology::Pyramid => sys::CEED_TOPOLOGY_PYRAMID,
            Topology::Prism => sys::CEED_TOPOLOGY_PRISM,
            Topology::Hexahedron => sys::CEED_TOPOLOGY_HEX,
        }
    }

    /// The dimension of the reference element.
    pub fn dim(self) -> usize {
        match self {
            Topology::Line => 1,
            Topology::Triangle | Topology::Quadrilateral => 2,
            _ => 3,
        }
    }

    /// The simplex of dimension `dim` (1, 2 or 3).
    fn simplex(dim: usize) -> Option<Topology> {
        match dim {
            1 => Some(Topology::Line),
            2 => Some(Topology::Triangle),
            3 => Some(Topology::Tetrahedron),
            _ => None,
        }
    }
}

/// A libCEED basis (`CeedBasis`).
#[derive(Debug)]
pub struct Basis {
    raw: sys::CeedBasis,
    ceed: Ceed,
    dim: usize,
    num_nodes: usize,
    num_qpts: usize,
}

impl Basis {
    /// An H1 basis of `num_nodes` nodes and `num_comp` components on `topology`, tabulated at
    /// `num_qpts` quadrature points (`CeedBasisCreateH1`, which copies the arrays). With `dim`
    /// the dimension of `topology`:
    /// - `interp[q * num_nodes + p]`: basis function `p` at point `q`;
    /// - `grad[(d * num_qpts + q) * num_nodes + p]`: its derivative in reference direction `d`;
    /// - `q_ref[d * num_qpts + q]`: coordinate `d` of point `q` on the reference element;
    /// - `q_weight[q]`: the weight of point `q`.
    #[allow(clippy::too_many_arguments)]
    pub fn h1(
        ceed: &Ceed,
        topology: Topology,
        num_comp: usize,
        num_nodes: usize,
        num_qpts: usize,
        interp: &[f64],
        grad: &[f64],
        q_ref: &[f64],
        q_weight: &[f64],
    ) -> Result<Basis> {
        let dim = topology.dim();
        for (name, len, expected) in [
            ("interp", interp.len(), num_qpts.checked_mul(num_nodes)),
            (
                "grad",
                grad.len(),
                num_qpts
                    .checked_mul(num_nodes)
                    .and_then(|n| n.checked_mul(dim)),
            ),
            ("q_ref", q_ref.len(), num_qpts.checked_mul(dim)),
            ("q_weight", q_weight.len(), Some(num_qpts)),
        ] {
            if Some(len) != expected {
                return Err(Error::new(
                    sys::CEED_ERROR_DIMENSION,
                    format!(
                        "basis array {name} has length {len}, expected {} for {num_nodes} nodes \
                         and {num_qpts} points in {dim}D",
                        expected.map_or("an overflowing size".to_string(), |e| e.to_string())
                    ),
                ));
            }
        }
        let mut raw: sys::CeedBasis = null_mut();
        // SAFETY: libCEED copies arrays of exactly the lengths checked above.
        chk!(ceed, unsafe {
            sys::CeedBasisCreateH1(
                ceed.as_raw(),
                topology.raw(),
                to_int(num_comp, "number of components")?,
                to_int(num_nodes, "number of nodes")?,
                to_int(num_qpts, "number of quadrature points")?,
                interp.as_ptr(),
                grad.as_ptr(),
                q_ref.as_ptr(),
                q_weight.as_ptr(),
                &mut raw,
            )
        });
        Ok(Basis {
            raw,
            ceed: ceed.clone(),
            dim,
            num_nodes,
            num_qpts,
        })
    }

    /// The Lagrange basis of `degree` with `num_comp` components on the reference simplex of
    /// dimension `dim` (1, 2 or 3), with a quadrature exact for polynomials of degree `q_order`.
    ///
    /// The reference simplex is the unit simplex, with vertices `0` and the unit vectors
    /// `e_1, .., e_dim` (volume `1 / dim!`). The nodes are the points `(i_1, .., i_dim) /
    /// degree` with `i_1 + .. + i_dim <= degree`, ordered lexicographically with `i_1` fastest:
    /// for `degree` 1 the vertices in the order above. The quadrature is the conical product
    /// (Stroud) rule of `ceil((q_order + dim) / 2)` Gauss-Legendre points per direction.
    pub fn h1_lagrange_simplex(
        ceed: &Ceed,
        dim: usize,
        num_comp: usize,
        degree: usize,
        q_order: usize,
    ) -> Result<Basis> {
        let topology = Topology::simplex(dim).ok_or_else(|| {
            Error::new(
                sys::CEED_ERROR_DIMENSION,
                format!("simplex of dimension {dim} (need 1, 2 or 3)"),
            )
        })?;
        if degree == 0 {
            return Err(Error::new(
                sys::CEED_ERROR_DIMENSION,
                "Lagrange basis of degree 0 (need at least 1)",
            ));
        }
        let (q_ref, q_weight) = simplex::quadrature(dim, q_order)?;
        let num_qpts = q_weight.len();
        let nodes = simplex::lattice(dim, degree);
        let num_nodes = nodes.len();
        let mut interp = vec![0.0; num_qpts * num_nodes];
        let mut grad = vec![0.0; dim * num_qpts * num_nodes];
        let mut x = vec![0.0; dim];
        let mut dphi = vec![0.0; dim];
        for q in 0..num_qpts {
            for (d, xd) in x.iter_mut().enumerate() {
                *xd = q_ref[d * num_qpts + q];
            }
            for (p, node) in nodes.iter().enumerate() {
                interp[q * num_nodes + p] = simplex::lagrange(degree, node, &x, &mut dphi);
                for (d, g) in dphi.iter().enumerate() {
                    grad[(d * num_qpts + q) * num_nodes + p] = *g;
                }
            }
        }
        Basis::h1(
            ceed, topology, num_comp, num_nodes, num_qpts, &interp, &grad, &q_ref, &q_weight,
        )
    }

    /// The dimension of the reference element.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// The number of quadrature points.
    pub fn num_quadrature_points(&self) -> usize {
        self.num_qpts
    }

    /// The quadrature weights on the reference element.
    pub fn quadrature_weights(&self) -> Result<Vec<f64>> {
        let mut ptr: *const f64 = null();
        chk!(&self.ceed, unsafe {
            sys::CeedBasisGetQWeights(self.raw, &mut ptr)
        });
        // SAFETY: the basis owns `num_qpts` weights.
        Ok(unsafe { slice::from_raw_parts(ptr, self.num_qpts) }.to_vec())
    }

    /// The raw handle, for the libCEED API not wrapped here.
    pub fn as_raw(&self) -> sys::CeedBasis {
        self.raw
    }

    pub(crate) fn ceed(&self) -> &Ceed {
        &self.ceed
    }
}

impl Drop for Basis {
    fn drop(&mut self) {
        // Nothing to report an error to; operators using it hold their own reference.
        unsafe { sys::CeedBasisDestroy(&mut self.raw) };
    }
}

/// Lagrange bases and quadrature on the unit simplex.
mod simplex {
    use crate::{Error, Result, sys};

    /// The lattice nodes `(i_1, .., i_dim)` with `sum <= degree`, `i_1` fastest.
    pub fn lattice(dim: usize, degree: usize) -> Vec<Vec<usize>> {
        if dim == 0 {
            return vec![vec![]];
        }
        // The last index slowest: for each value of it, the lattice of the lower dimensions.
        let mut nodes = Vec::new();
        for last in 0..=degree {
            for mut lower in lattice(dim - 1, degree - last) {
                lower.push(last);
                nodes.push(lower);
            }
        }
        nodes
    }

    /// `prod_{m < a} (p t - m) / (m + 1)` and its derivative in `t`.
    fn factor(p: usize, a: usize, t: f64) -> (f64, f64) {
        let pt = p as f64 * t;
        let (mut value, mut deriv) = (1.0, 0.0);
        for m in 0..a {
            let (f, df) = ((pt - m as f64) / (m + 1) as f64, p as f64 / (m + 1) as f64);
            deriv = deriv * f + value * df;
            value *= f;
        }
        (value, deriv)
    }

    /// The Lagrange function of the lattice node `node` of `degree` at `x`, and its gradient
    /// into `grad`: the product over the barycentric coordinates `l_0 = 1 - sum x`, `l_k =
    /// x_k` of `factor(degree, a_k, l_k)`, with `a_0 = degree - sum node`, `a_k = node_k`.
    pub fn lagrange(degree: usize, node: &[usize], x: &[f64], grad: &mut [f64]) -> f64 {
        let dim = x.len();
        let a0 = degree - node.iter().sum::<usize>();
        let l0 = 1.0 - x.iter().sum::<f64>();
        let (f0, df0) = factor(degree, a0, l0);
        let factors: Vec<(f64, f64)> = (0..dim).map(|k| factor(degree, node[k], x[k])).collect();
        let prod_except = |skip: Option<usize>| {
            factors
                .iter()
                .enumerate()
                .filter(|&(k, _)| Some(k) != skip)
                .map(|(_, f)| f.0)
                .product::<f64>()
        };
        let all = prod_except(None);
        for (j, g) in grad.iter_mut().enumerate() {
            // d l_0 / d x_j = -1, d l_k / d x_j = delta_kj.
            *g = f0 * factors[j].1 * prod_except(Some(j)) - df0 * all;
        }
        f0 * all
    }

    /// Gauss-Legendre points and weights on [0, 1].
    fn gauss(n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
        let (mut x, mut w) = (vec![0.0; n], vec![0.0; n]);
        let q = crate::to_int(n, "number of Gauss points")?;
        // Does not fail (no checks, no allocation); writes n values each.
        let ierr = unsafe { sys::CeedGaussQuadrature(q, x.as_mut_ptr(), w.as_mut_ptr()) };
        if ierr != 0 {
            return Err(Error::new(ierr, "CeedGaussQuadrature failed"));
        }
        Ok((
            x.iter().map(|x| (x + 1.0) / 2.0).collect(),
            w.iter().map(|w| w / 2.0).collect(),
        ))
    }

    /// The conical product rule on the unit simplex of `dim`, exact to degree `order`: points
    /// (`q_ref[d * n + q]`) and weights. The collapsed coordinates `t` in [0, 1]^dim map to
    /// `x_k = t_k prod_{j < k} (1 - t_j)`, with the Jacobian `prod_j (1 - t_j)^(dim - 1 - j)`,
    /// which raises the degree in `t_0` by `dim - 1`.
    pub fn quadrature(dim: usize, order: usize) -> Result<(Vec<f64>, Vec<f64>)> {
        let n1 = (order + dim).div_ceil(2);
        let (t, w) = gauss(n1)?;
        let n = n1.pow(dim as u32);
        let mut q_ref = vec![0.0; dim * n];
        let mut q_weight = vec![0.0; n];
        for (q, weight) in q_weight.iter_mut().enumerate() {
            // Digit j of q in base n1 is the point in collapsed direction j.
            let mut rest = q;
            let mut scale = 1.0;
            *weight = 1.0;
            for j in 0..dim {
                let i = rest % n1;
                rest /= n1;
                q_ref[j * n + q] = t[i] * scale;
                *weight *= w[i] * (1.0 - t[i]).powi((dim - 1 - j) as i32);
                scale *= 1.0 - t[i];
            }
        }
        Ok((q_ref, q_weight))
    }
}
