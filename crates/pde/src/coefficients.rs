//! Coefficients at the quadrature points of a [`Discretization`]: the values of the coefficients
//! of a study ([`coefficient_values`]), and the data of the [`crate::Transport`] operator
//! ([`PointData`]).

use std::cell::OnceCell;

use ceed::{CVector, Ceed};
use contracts::Coefficient;
use maps::{Map, MapError, Maps};
use petsc::sys::PetscInt;

use crate::qfunctions::TRANSPORT_DATA_SIZE;
use crate::{Discretization, PdeError, Result};

/// The values of the coefficient `c` at the quadrature points of the local cells of `disc`, in
/// the layout of [`Discretization::quadrature_points`]: the constant itself, or the table
/// evaluated at the value of its map there ([`Coefficient::eval`] of [`maps::Map::sample`]).
/// `maps` holds the map of a table, read for the cells of this process. Collective.
///
/// Errors (on every process): [`PdeError::Map`] if the map of a table is not in `maps`, or a
/// point is outside the map or on a no-value pixel. The message is the error of the first
/// such point of the lowest process which has one: the map, the point and the area tag of its
/// cell ([`Discretization::cell_area_tag`]); and the number of points which fail, on all
/// processes.
pub fn coefficient_values(disc: &Discretization, c: &Coefficient, maps: &Maps) -> Result<Vec<f64>> {
    match c {
        Coefficient::Constant(value) => Ok(vec![*value; disc.num_quadrature_points()]),
        Coefficient::Table(table) => map_values(disc, &table.map, maps, f64::NAN, |map, x| {
            map.sample(x).map(|value| c.eval(value))
        }),
    }
}

/// The gradient (per um, [`maps::Map::gradient`]) of the map `id` at the quadrature points of
/// the local cells of `disc`, in the layout of [`Discretization::quadrature_points`]. `maps`
/// holds the map, read for the cells of this process. Collective.
///
/// Errors (on every process): [`PdeError::Map`] as for [`coefficient_values`].
pub(crate) fn map_gradients(disc: &Discretization, id: &str, maps: &Maps) -> Result<Vec<[f64; 2]>> {
    map_values(disc, id, maps, [f64::NAN; 2], |map, x| map.gradient(x))
}

/// `f(map, x)` of the map `id` at the quadrature points `x` of the local cells of `disc`, in the
/// layout of [`Discretization::quadrature_points`], with `failed` at the points where `f` fails.
/// Collective.
///
/// Errors (on every process): [`PdeError::Map`] if the map is not in `maps` (on any process), or
/// `f` fails at a point of any process: the error of the first such point of the lowest process
/// which has one, with the area tag of its cell, and the number of points which fail on all
/// processes.
fn map_values<T: Copy>(
    disc: &Discretization,
    id: &str,
    maps: &Maps,
    failed: T,
    f: impl Fn(&Map, [f64; 2]) -> Result<T, MapError>,
) -> Result<Vec<T>> {
    let points = disc.quadrature_points();
    let comm = disc.dm().comm();
    let map = maps.get(id);
    // A map one process lacks: the processes read the same maps, so all lack it.
    let missing = comm.max_int(map.is_err().into())? > 0;
    let map = match map {
        Ok(map) if !missing => map,
        Ok(_) => {
            return Err(PdeError::Map {
                message: format!("map '{id}' is missing on another process"),
            });
        }
        Err(e) => {
            return Err(PdeError::Map {
                message: e.to_string(),
            });
        }
    };

    let q = disc.points_per_cell();
    let mut values = Vec::with_capacity(points.len());
    // The first point which fails, and the number of them.
    let mut first: Option<(usize, MapError)> = None;
    let mut failing: usize = 0;
    for (i, &x) in points.iter().enumerate() {
        match f(map, x) {
            Ok(value) => values.push(value),
            Err(e) => {
                values.push(failed);
                failing += 1;
                first.get_or_insert((i, e));
            }
        }
    }
    let failure = first.map(|(i, e)| {
        let tag = disc.cell_area_tag_index(i / q);
        PointFailure::new(e, tag)
    });
    match PointFailure::reduce(failure, failing, &comm)? {
        None => Ok(values),
        Some((failure, count)) => Err(failure.into_error(id, disc, count)),
    }
}

/// A map error at a point, in numbers which travel between processes by reductions.
#[derive(Debug, Clone, Copy, PartialEq)]
struct PointFailure {
    /// 1: outside the raster, 2: a no-value pixel.
    kind: PetscInt,
    x: [f64; 2],
    /// The index of the area tag of the cell (see [`Discretization::cell_area_tag_index`]), or
    /// -1.
    tag: PetscInt,
}

const OUTSIDE: PetscInt = 1;
const NO_VALUE: PetscInt = 2;

impl PointFailure {
    fn new(e: MapError, tag: Option<usize>) -> PointFailure {
        let (kind, x) = match e {
            MapError::Outside { x, .. } => (OUTSIDE, x),
            MapError::NoValue { x, .. } => (NO_VALUE, x),
            // Sampling fails at a point, nothing else (and without a point, NaN).
            _ => (OUTSIDE, [f64::NAN; 2]),
        };
        let tag = tag.and_then(|t| PetscInt::try_from(t).ok()).unwrap_or(-1);
        PointFailure { kind, x, tag }
    }

    /// The failure of the lowest process which has one, with the number of points which fail on
    /// all processes (`failing` on this one); `None` on every process if none fails.
    /// Collective.
    fn reduce(
        local: Option<PointFailure>,
        failing: usize,
        comm: &petsc::Comm,
    ) -> Result<Option<(PointFailure, usize)>> {
        let rank = PetscInt::try_from(comm.rank()).expect("a rank is a PETSc integer");
        // The lowest rank which fails, as the maximum of its negation.
        let key = comm.max_int(if local.is_some() {
            -rank
        } else {
            PetscInt::MIN
        })?;
        if key == PetscInt::MIN {
            return Ok(None);
        }
        let mine = if -key == rank { local } else { None };
        // Every other process adds nothing: 0 to the integers, 0.0 to the coordinates.
        let failure = mine.unwrap_or(PointFailure {
            kind: 0,
            x: [0.0; 2],
            tag: -1,
        });
        let count = PetscInt::try_from(failing).unwrap_or(PetscInt::MAX);
        let failure = PointFailure {
            kind: comm.max_int(failure.kind)?,
            x: [comm.sum_real(failure.x[0])?, comm.sum_real(failure.x[1])?],
            tag: comm.max_int(failure.tag)?,
        };
        let count = comm.sum_int(count)?;
        Ok(Some((
            failure,
            usize::try_from(count).unwrap_or(usize::MAX),
        )))
    }

    /// The error of the failure at a point of `disc`, of the map `map`, with `count` points which
    /// fail in all.
    fn into_error(self, map: &str, disc: &Discretization, count: usize) -> PdeError {
        let error = if self.kind == NO_VALUE {
            MapError::NoValue {
                map: map.to_string(),
                x: self.x,
            }
        } else {
            MapError::Outside {
                map: map.to_string(),
                x: self.x,
            }
        };
        let tag = usize::try_from(self.tag)
            .ok()
            .and_then(|t| disc.shared().area_tags.get(t))
            .map_or_else(
                || "in a cell without area tag".to_string(),
                |t| format!("in area tag '{t}'"),
            );
        let points = if count == 1 { "point" } else { "points" };
        PdeError::Map {
            message: format!(
                "{error}, {tag}; the map fails at {count} quadrature {points} of the mesh"
            ),
        }
    }
}

/// The coefficients of the [`crate::Transport`] operator at the quadrature points of a
/// discretization, in the layout of [`Discretization::quadrature_points`]:
///
/// - `mass`: the factor of `u_t`, for example the porosity `phi`;
/// - `reaction`: the factor of `u`, for example `phi k` for the decay rate `k` (1/s);
/// - `diffusion`: the symmetric diffusion tensor `D` as `[D00, D01, D11]`, for example
///   `phi d I` (um^2/s);
/// - `velocity` (um/s) and `tau` (s), the SUPG parameter: for advection, zero unless set
///   ([`PointData::with_velocity`], [`PointData::stabilized`]).
///
/// The operators of a discretization may share it (as an `Rc`): they share the copy libCEED
/// reads too.
#[derive(Debug)]
pub struct PointData {
    mass: Vec<f64>,
    reaction: Vec<f64>,
    diffusion: Vec<[f64; 3]>,
    velocity: Vec<[f64; 2]>,
    tau: Vec<f64>,
    /// The values as libCEED reads them, and its context: made by the first operator.
    packed: OnceCell<(Ceed, CVector)>,
}

impl PointData {
    /// The data of the coefficients `mass`, `reaction` and `diffusion` (see [`PointData`]), one
    /// value per point, without velocity: velocity and `tau` zero.
    ///
    /// Panics unless the three have the same length.
    pub fn new(mass: Vec<f64>, reaction: Vec<f64>, diffusion: Vec<[f64; 3]>) -> PointData {
        let n = mass.len();
        assert!(
            reaction.len() == n && diffusion.len() == n,
            "point data of different lengths: mass {n}, reaction {}, diffusion {}",
            reaction.len(),
            diffusion.len()
        );
        PointData {
            mass,
            reaction,
            diffusion,
            velocity: vec![[0.0; 2]; n],
            tau: vec![0.0; n],
            packed: OnceCell::new(),
        }
    }

    /// The data of an isotropic diffusion coefficient: the tensor `diffusion I` at each point
    /// of `disc`, with `mass` and `reaction` (see [`PointData::new`]).
    ///
    /// Panics unless each has one value per quadrature point of `disc`.
    pub fn isotropic(
        disc: &Discretization,
        mass: &[f64],
        reaction: &[f64],
        diffusion: &[f64],
    ) -> PointData {
        let n = disc.num_quadrature_points();
        assert!(
            mass.len() == n && reaction.len() == n && diffusion.len() == n,
            "point data of lengths {}, {}, {} for {n} quadrature points",
            mass.len(),
            reaction.len(),
            diffusion.len()
        );
        let diffusion = diffusion.iter().map(|&d| [d, 0.0, d]).collect();
        PointData::new(mass.to_vec(), reaction.to_vec(), diffusion)
    }

    /// The data with the velocity `velocity` (um/s) at the points, and `tau` zero: the Galerkin
    /// form of advection, unless [`PointData::stabilized`].
    ///
    /// Panics unless it has one value per point.
    pub fn with_velocity(mut self, velocity: Vec<[f64; 2]>) -> PointData {
        assert_eq!(velocity.len(), self.len(), "velocities for the points");
        self.velocity = velocity;
        self.tau = vec![0.0; self.len()];
        self.packed = OnceCell::new();
        self
    }

    /// The data with the SUPG parameter `tau` (s) at the points of `disc`, from the velocity
    /// `u`, the diffusion tensor `D` and the reaction `r` there (Shakib, Hughes and Johan 1991,
    /// for the steady terms, with the reaction as in Codina 2000):
    ///
    /// `tau = ((2 |u| / h)^2 + 9 (4 d_u / h^2)^2 + r^2)^(-1/2)`,
    ///
    /// `d_u = u . D u / |u|^2` the diffusion along the flow, and `h` the length of the cell
    /// along the flow, `h = 2 |u| / sum_i |u . grad N_i|` over the P1 basis functions `N_i` of
    /// the cell (Tezduyar's `h_UGN`; for a triangle the longest chord parallel to `u`). `tau` is
    /// zero where the velocity is zero. Its limits are the classical ones: `h / (2 |u|)` for
    /// pure advection, `h^2 / (12 d_u)` where diffusion dominates.
    ///
    /// The transient term of SUPG (the porosity times `u_t` in the residual its test function
    /// takes) is in the operators ([`crate::Terms::Mass`]); `tau` itself leaves out the time
    /// step (the `(2 / dt)^2` of Shakib, Hughes and Johan): it would change with every step of
    /// an adaptive integrator, and make the stabilization vanish for small steps, which brings
    /// back the oscillations SUPG removes.
    ///
    /// Panics unless the data has one value per quadrature point of `disc`.
    pub fn stabilized(mut self, disc: &Discretization) -> PointData {
        let n = disc.num_quadrature_points();
        assert_eq!(self.len(), n, "point data for the quadrature points");
        let inverse = disc
            .inverse_jacobians()
            .expect("the quadrature data of a discretization is readable");
        let gradients = disc.reference_gradients();
        self.tau = (0..n)
            .map(|p| {
                let [u0, u1] = self.velocity[p];
                let speed2 = u0 * u0 + u1 * u1;
                if speed2 == 0.0 {
                    return 0.0;
                }
                // u . grad N_i = sum_c u_c sum_j dN_i/dX_j b_jc = dN_i/dX . w, w_j = b_jc u_c.
                let [[b00, b01], [b10, b11]] = inverse[p];
                let w = [b00 * u0 + b01 * u1, b10 * u0 + b11 * u1];
                // 2 |u| / h.
                let rate: f64 = gradients
                    .iter()
                    .map(|g| (g[0] * w[0] + g[1] * w[1]).abs())
                    .sum();
                let h = 2.0 * speed2.sqrt() / rate;
                let [d00, d01, d11] = self.diffusion[p];
                let along = (u0 * (d00 * u0 + d01 * u1) + u1 * (d01 * u0 + d11 * u1)) / speed2;
                let diffusion = 4.0 * along / (h * h);
                let r = self.reaction[p];
                1.0 / (rate * rate + 9.0 * diffusion * diffusion + r * r).sqrt()
            })
            .collect();
        self.packed = OnceCell::new();
        self
    }

    /// The number of points.
    pub fn len(&self) -> usize {
        self.mass.len()
    }

    /// Whether there are no points (a process without cells).
    pub fn is_empty(&self) -> bool {
        self.mass.is_empty()
    }

    /// The factor of `u_t` at the points.
    pub fn mass(&self) -> &[f64] {
        &self.mass
    }

    /// The factor of `u` at the points.
    pub fn reaction(&self) -> &[f64] {
        &self.reaction
    }

    /// The diffusion tensor at the points, `[D00, D01, D11]`.
    pub fn diffusion(&self) -> &[[f64; 3]] {
        &self.diffusion
    }

    /// The velocity at the points.
    pub fn velocity(&self) -> &[[f64; 2]] {
        &self.velocity
    }

    /// The SUPG parameter at the points.
    pub fn tau(&self) -> &[f64] {
        &self.tau
    }

    /// Calls `f` with the values as `Transport2D` reads them (`TRANSPORT_DATA_SIZE`
    /// components per point, in the layout `[cell][component][point]` of the strided
    /// restrictions of the discretization) for cells of `q` points, on `ceed`. They are made
    /// once, on the libCEED context of the first call, and shared by the operators (libCEED
    /// keeps a reference to the vector of an operator field); a call with another context gets
    /// a copy of its own.
    pub(crate) fn with_packed<R>(
        &self,
        ceed: &Ceed,
        q: usize,
        f: impl FnOnce(&CVector) -> Result<R>,
    ) -> Result<R> {
        match self.packed.get() {
            Some((on, packed)) if on.as_raw() == ceed.as_raw() => f(packed),
            Some(_) => f(&self.pack(ceed, q)?),
            None => {
                let packed = self.pack(ceed, q)?;
                let (_, packed) = self.packed.get_or_init(|| (ceed.clone(), packed));
                f(packed)
            }
        }
    }

    fn pack(&self, ceed: &Ceed, q: usize) -> Result<CVector> {
        let n = self.len();
        if q == 0 || !n.is_multiple_of(q) {
            return Err(PdeError::Unsupported(format!(
                "point data of {n} points for cells of {q} points"
            )));
        }
        let mut values = vec![0.0; n * TRANSPORT_DATA_SIZE];
        for (cell, out) in values.chunks_exact_mut(TRANSPORT_DATA_SIZE * q).enumerate() {
            for i in 0..q {
                let p = cell * q + i;
                let [d00, d01, d11] = self.diffusion[p];
                let [u0, u1] = self.velocity[p];
                let point = [
                    self.mass[p],
                    self.reaction[p],
                    d00,
                    d01,
                    d11,
                    u0,
                    u1,
                    self.tau[p],
                ];
                for (component, value) in point.into_iter().enumerate() {
                    out[component * q + i] = value;
                }
            }
        }
        Ok(CVector::from_slice(ceed, &values)?)
    }
}
