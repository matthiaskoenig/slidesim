//! Spike (throwaway): mesh a tagged 2D domain with `spade` (constrained Delaunay) and remesh it
//! with PETSc's `DMAdaptMetric` and Mmg to the metric of a size field, natively and in
//! wasm32-wasip1. Prints counts, quality (min angle, min SICN, edge ratios), per-tag lengths,
//! a hash of the output mesh and the wall times; writes a VTU (`-vtu <path>`).
//!
//! Fixtures (`-fixture lobule|split`):
//! - `lobule`: one lobule, a hexagon of circumradius 300 um (tag `field_cut`) with a central
//!   hole of 30 um (`central_wall`) and six corner holes of 25 um (`portal_wall`), circles
//!   polygonized at 5 um; one area.
//! - `split`: a 400 um square (`field_cut`) split by an interior polyline (`necrosis_border`)
//!   into two areas, with one hole of 25 um (`vessel_wall`) in the lower area.
//!
//! Size field (gmsh's Threshold field, slideviz's "5:20"): `h = 5` up to 10 um from the nearest
//! wall (every tagged chain but `field_cut`), growing linearly to `h = 20` at 200 um, 20 beyond.
//!
//! Steps: the input chains are split to the size field; `spade` triangulates them with every
//! sub-edge as a constraint and refines lightly (sub-edges kept, `-refine_angle`, `-refine_area`);
//! triangles outside the domain are dropped; `DMPlexCreateFromCellListPetsc`; one combined
//! edge label `walls` (value = tag index + 1) and one cell label `areas` (area index + 1); an
//! isotropic metric `1/h^2` per vertex (`DMPlexMetricCreateIsotropic`); `DMAdaptMetric` with
//! `-dm_adaptor mmg -dm_plex_metric_no_surf` (frozen boundary and interface edges, the
//! defaults set here unless given on the command line); the labels are split back into one
//! label per tag.

mod ffi {
    #![allow(
        non_upper_case_globals,
        non_camel_case_types,
        non_snake_case,
        dead_code
    )]
    #![allow(clippy::all, unsafe_op_in_unsafe_fn, improper_ctypes, unused_imports)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_int};
use std::fmt::Write as _;
use std::ptr::{null, null_mut};
use std::time::Instant;

use spade::{
    AngleLimit, ConstrainedDelaunayTriangulation, Point2, RefinementParameters, Triangulation,
};

// `Vec` below is PETSc's vector (the glob import shadows the prelude); Rust vectors are written
// `std::vec::Vec`.
use ffi::*;

macro_rules! petsc {
    ($e:expr) => {{
        let ierr = unsafe { $e };
        if ierr != 0 {
            panic!("PETSc error {} in `{}`", ierr, stringify!($e));
        }
    }};
}

type P = [f64; 2];

const PI: f64 = std::f64::consts::PI;

/// A tagged input polyline (closed: the last point connects to the first).
struct Chain {
    pts: std::vec::Vec<P>,
    closed: bool,
    tag: usize,
}

impl Chain {
    fn segments(&self) -> impl Iterator<Item = (P, P)> + '_ {
        let n = self.pts.len();
        let m = if self.closed { n } else { n - 1 };
        (0..m).map(move |i| (self.pts[i], self.pts[(i + 1) % n]))
    }
}

struct Geometry {
    tags: &'static [&'static str],
    areas: &'static [&'static str],
    chains: std::vec::Vec<Chain>,
    /// Area polygons (outer rings, in area order); a point belongs to the first containing it.
    area_polys: std::vec::Vec<std::vec::Vec<P>>,
    holes: std::vec::Vec<std::vec::Vec<P>>,
    /// Wall segments of the size field (every chain but `field_cut`, tag 0).
    walls: std::vec::Vec<(P, P)>,
}

/// A circle of radius `r` around `c`, polygonized at (at most) `h`, counterclockwise.
fn circle(c: P, r: f64, h: f64) -> std::vec::Vec<P> {
    let n = (2.0 * PI * r / h).ceil() as usize;
    (0..n)
        .map(|k| {
            let a = 2.0 * PI * k as f64 / n as f64;
            [c[0] + r * libm::cos(a), c[1] + r * libm::sin(a)]
        })
        .collect()
}

fn lobule() -> Geometry {
    let (radius, central, portal, wall_h) = (300.0, 30.0, 25.0, 5.0);
    let hexagon: std::vec::Vec<P> = (0..6)
        .map(|k| {
            let a = PI / 3.0 * k as f64;
            [radius * libm::cos(a), radius * libm::sin(a)]
        })
        .collect();
    let mut chains = vec![
        Chain {
            pts: hexagon.clone(),
            closed: true,
            tag: 0,
        },
        Chain {
            pts: circle([0.0, 0.0], central, wall_h),
            closed: true,
            tag: 1,
        },
    ];
    // Portal holes towards the six corners, 250 um from the center (their walls 18 um from
    // the hexagon's edges): holes, not notches cut by the field_cut.
    for k in 0..6 {
        let a = PI / 3.0 * k as f64;
        let c = [250.0 * libm::cos(a), 250.0 * libm::sin(a)];
        chains.push(Chain {
            pts: circle(c, portal, wall_h),
            closed: true,
            tag: 2,
        });
    }
    finish(
        &["field_cut", "central_wall", "portal_wall"],
        &["parenchyma"],
        chains,
        vec![hexagon],
    )
}

fn split() -> Geometry {
    let s = 200.0;
    // Interior chain from the left to the right side (the necrosis border), kinked.
    let border: std::vec::Vec<P> = vec![
        [-s, -20.0],
        [-80.0, 40.0],
        [0.0, -10.0],
        [90.0, 50.0],
        [s, 10.0],
    ];
    // The outer ring as one closed chain (it contains both ends of the border as vertices).
    let outer = vec![[-s, -s], [s, -s], border[4], [s, s], [-s, s], border[0]];
    let mut upper: std::vec::Vec<P> = border.clone();
    upper.extend([[s, s], [-s, s]]);
    let mut lower: std::vec::Vec<P> = border.iter().rev().copied().collect();
    lower.extend([[-s, -s], [s, -s]]);
    let hole = circle([0.0, -110.0], 25.0, 5.0);
    let chains = vec![
        Chain {
            pts: outer,
            closed: true,
            tag: 0,
        },
        Chain {
            pts: border,
            closed: false,
            tag: 1,
        },
        Chain {
            pts: hole,
            closed: true,
            tag: 2,
        },
    ];
    finish(
        &["field_cut", "necrosis_border", "vessel_wall"],
        &["upper", "lower"],
        chains,
        vec![upper, lower],
    )
}

fn finish(
    tags: &'static [&'static str],
    areas: &'static [&'static str],
    chains: std::vec::Vec<Chain>,
    area_polys: std::vec::Vec<std::vec::Vec<P>>,
) -> Geometry {
    let holes = chains
        .iter()
        .filter(|c| c.closed && c.tag != 0)
        .map(|c| c.pts.clone())
        .collect();
    let walls = chains
        .iter()
        .filter(|c| c.tag != 0)
        .flat_map(|c| c.segments().collect::<std::vec::Vec<_>>())
        .collect();
    Geometry {
        tags,
        areas,
        chains,
        area_polys,
        holes,
        walls,
    }
}

fn sub(a: P, b: P) -> P {
    [a[0] - b[0], a[1] - b[1]]
}
fn dot(a: P, b: P) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}
fn cross(a: P, b: P) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}
fn norm(a: P) -> f64 {
    dot(a, a).sqrt()
}
fn dist(a: P, b: P) -> f64 {
    norm(sub(a, b))
}

/// Distance of `p` to the segment `ab`.
fn seg_dist(p: P, a: P, b: P) -> f64 {
    let ab = sub(b, a);
    let t = (dot(sub(p, a), ab) / dot(ab, ab)).clamp(0.0, 1.0);
    dist(p, [a[0] + t * ab[0], a[1] + t * ab[1]])
}

fn inside(p: P, poly: &[P]) -> bool {
    let mut c = false;
    let n = poly.len();
    for i in 0..n {
        let (a, b) = (poly[i], poly[(i + 1) % n]);
        if (a[1] > p[1]) != (b[1] > p[1]) {
            let x = a[0] + (p[1] - a[1]) / (b[1] - a[1]) * (b[0] - a[0]);
            if p[0] < x {
                c = !c;
            }
        }
    }
    c
}

impl Geometry {
    /// The size field: 5 um up to 10 um from the nearest wall, then linear to 20 um at 200 um.
    fn h(&self, p: P) -> f64 {
        let d = self
            .walls
            .iter()
            .map(|&(a, b)| seg_dist(p, a, b))
            .fold(f64::INFINITY, f64::min);
        let (h_min, h_max, d_min, d_max) = (5.0, 20.0, 10.0, 200.0);
        if d <= d_min {
            h_min
        } else if d >= d_max {
            h_max
        } else {
            h_min + (h_max - h_min) * (d - d_min) / (d_max - d_min)
        }
    }

    fn area_of(&self, p: P) -> Option<usize> {
        if self.holes.iter().any(|h| inside(p, h)) {
            return None;
        }
        self.area_polys.iter().position(|a| inside(p, a))
    }

    /// Input length per tag (the polylines as given, before splitting).
    fn tag_lengths(&self) -> std::vec::Vec<f64> {
        let mut len = vec![0.0; self.tags.len()];
        for c in &self.chains {
            for (a, b) in c.segments() {
                len[c.tag] += dist(a, b);
            }
        }
        len
    }

    /// The parameters `t` (0 and 1 included) splitting the segment `ab` to the size field:
    /// `n = round(int_0^L ds / h)` sub-edges of equal `int ds / h`.
    fn split_params(&self, a: P, b: P) -> std::vec::Vec<f64> {
        const M: usize = 256;
        let l = dist(a, b);
        let at = |t: f64| [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
        let inv: std::vec::Vec<f64> = (0..=M)
            .map(|i| 1.0 / self.h(at(i as f64 / M as f64)))
            .collect();
        let mut g = vec![0.0; M + 1];
        for i in 0..M {
            g[i + 1] = g[i] + 0.5 * (inv[i] + inv[i + 1]) * l / M as f64;
        }
        let n = (g[M].round() as usize).max(1);
        let mut ts = vec![0.0];
        let mut i = 0;
        for k in 1..n {
            let target = g[M] * k as f64 / n as f64;
            while g[i + 1] < target {
                i += 1;
            }
            let f = (target - g[i]) / (g[i + 1] - g[i]);
            ts.push((i as f64 + f) / M as f64);
        }
        ts.push(1.0);
        ts
    }
}

/// The triangulation handed to PETSc: vertices, triangles (counterclockwise), tagged edges
/// (vertex pairs) and the area of every triangle.
struct Input {
    coords: std::vec::Vec<P>,
    cells: std::vec::Vec<[usize; 3]>,
    edges: std::vec::Vec<([usize; 2], usize)>,
    cell_area: std::vec::Vec<usize>,
}

fn triangulate(geo: &Geometry, refine_angle: f64, refine_area: f64, exclude: bool) -> Input {
    let mut cdt = ConstrainedDelaunayTriangulation::<Point2<f64>>::new();
    // Constraint sub-edges with their tags, by spade vertex index.
    let mut constraints: std::vec::Vec<([usize; 2], usize)> = std::vec::Vec::new();
    for c in &geo.chains {
        for (a, b) in c.segments() {
            let ts = geo.split_params(a, b);
            let pts: std::vec::Vec<P> = ts
                .iter()
                .map(|&t| {
                    if t == 0.0 {
                        a
                    } else if t == 1.0 {
                        b
                    } else {
                        [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
                    }
                })
                .collect();
            let handles: std::vec::Vec<_> = pts
                .iter()
                .map(|p| cdt.insert(Point2::new(p[0], p[1])).expect("insert"))
                .collect();
            for w in handles.windows(2) {
                let crossing = cdt.add_constraint(w[0], w[1]);
                assert!(crossing, "constraint edge intersects another one");
                constraints.push(([w[0].index(), w[1].index()], c.tag));
            }
        }
    }
    if refine_angle > 0.0 || refine_area > 0.0 {
        let mut params = RefinementParameters::<f64>::new()
            .exclude_outer_faces(exclude)
            .keep_constraint_edges()
            .with_max_additional_vertices(200_000);
        if refine_angle > 0.0 {
            params = params.with_angle_limit(AngleLimit::from_deg(refine_angle));
        } else {
            params = params.with_angle_limit(AngleLimit::from_deg(0.0));
        }
        if refine_area > 0.0 {
            params = params.with_max_allowed_area(refine_area);
        }
        cdt.refine(params);
    }
    // Regions: faces connected across non-constraint edges (flood fill). A region touching the
    // outer face (across a convex hull edge) is outside; any other region gets the area of the
    // centroid of its largest face (not of every face: the slivers between nearly collinear
    // boundary sub-edges and the convex hull have centroids on the boundary).
    let nf = cdt.num_all_faces();
    let mut region = vec![usize::MAX; nf];
    let mut regions: std::vec::Vec<(bool, f64, P)> = std::vec::Vec::new(); // (outside, max area, its centroid)
    for f in cdt.inner_faces() {
        if region[f.fix().index()] != usize::MAX {
            continue;
        }
        let r = regions.len();
        regions.push((false, -1.0, [0.0, 0.0]));
        let mut stack = vec![f.fix()];
        region[f.fix().index()] = r;
        while let Some(fh) = stack.pop() {
            let face = cdt.face(fh);
            let v = face.vertices();
            let p: [P; 3] = std::array::from_fn(|i| [v[i].position().x, v[i].position().y]);
            let a = 0.5 * cross(sub(p[1], p[0]), sub(p[2], p[0])).abs();
            if a > regions[r].1 {
                regions[r].1 = a;
                regions[r].2 = [
                    (p[0][0] + p[1][0] + p[2][0]) / 3.0,
                    (p[0][1] + p[1][1] + p[2][1]) / 3.0,
                ];
            }
            for e in face.adjacent_edges() {
                if e.is_constraint_edge() {
                    continue;
                }
                match e.rev().face().as_inner() {
                    None => regions[r].0 = true,
                    Some(n) => {
                        if region[n.fix().index()] == usize::MAX {
                            region[n.fix().index()] = r;
                            stack.push(n.fix());
                        }
                    }
                }
            }
        }
    }
    let region_area: std::vec::Vec<Option<usize>> = regions
        .iter()
        .map(|&(outside, _, c)| if outside { None } else { geo.area_of(c) })
        .collect();
    // Triangles inside the domain, vertices renumbered in first use order.
    let mut map: HashMap<usize, usize> = HashMap::new();
    let mut coords = std::vec::Vec::new();
    let mut cells = std::vec::Vec::new();
    let mut cell_area = std::vec::Vec::new();
    for f in cdt.inner_faces() {
        let Some(area) = region_area[region[f.fix().index()]] else {
            continue;
        };
        let v = f.vertices();
        let p: [P; 3] = std::array::from_fn(|i| {
            let q = v[i].position();
            [q.x, q.y]
        });
        let tri: [usize; 3] = std::array::from_fn(|i| {
            *map.entry(v[i].fix().index()).or_insert_with(|| {
                coords.push(p[i]);
                coords.len() - 1
            })
        });
        cells.push(tri);
        cell_area.push(area);
    }
    let edges = constraints
        .iter()
        .map(|&([a, b], tag)| ([map[&a], map[&b]], tag))
        .collect();
    Input {
        coords,
        cells,
        edges,
        cell_area,
    }
}

fn opt_string(name: &CStr) -> Option<String> {
    let mut buf = [0 as c_char; 4096];
    let mut set = false;
    petsc!(PetscOptionsGetString(
        null_mut(),
        null(),
        name.as_ptr(),
        buf.as_mut_ptr(),
        buf.len(),
        &mut set
    ));
    set.then(|| {
        unsafe { CStr::from_ptr(buf.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    })
}

fn opt_real(name: &CStr, default: f64) -> f64 {
    let mut v = default;
    let mut set = false;
    petsc!(PetscOptionsGetReal(
        null_mut(),
        null(),
        name.as_ptr(),
        &mut v,
        &mut set
    ));
    v
}

fn default_option(name: &CStr, value: &CStr) {
    let mut set = false;
    petsc!(PetscOptionsHasName(
        null_mut(),
        null(),
        name.as_ptr(),
        &mut set
    ));
    if !set {
        petsc!(PetscOptionsSetValue(
            null_mut(),
            name.as_ptr(),
            value.as_ptr()
        ));
    }
}

/// The output mesh read back from the DMPlex.
struct Output {
    coords: std::vec::Vec<P>,
    cells: std::vec::Vec<[usize; 3]>,
    /// Every edge with its `walls` value (-1 untagged).
    edges: std::vec::Vec<([usize; 2], i64)>,
    cell_area: std::vec::Vec<i64>,
}

fn read_dm(dm: DM, walls: &CStr, areas: &CStr) -> Output {
    let (mut c0, mut c1, mut e0, mut e1, mut v0, mut v1) = (0, 0, 0, 0, 0, 0);
    petsc!(DMPlexGetHeightStratum(dm, 0, &mut c0, &mut c1));
    petsc!(DMPlexGetHeightStratum(dm, 1, &mut e0, &mut e1));
    petsc!(DMPlexGetDepthStratum(dm, 0, &mut v0, &mut v1));
    let mut csec: PetscSection = null_mut();
    let mut cvec: Vec = null_mut();
    petsc!(DMGetCoordinateSection(dm, &mut csec));
    petsc!(DMGetCoordinatesLocal(dm, &mut cvec));
    let mut arr: *const PetscScalar = null();
    petsc!(VecGetArrayRead(cvec, &mut arr));
    let coords = (v0..v1)
        .map(|v| {
            let mut off = 0;
            petsc!(PetscSectionGetOffset(csec, v, &mut off));
            unsafe { [*arr.add(off as usize), *arr.add(off as usize + 1)] }
        })
        .collect();
    petsc!(VecRestoreArrayRead(cvec, &mut arr));
    let mut wl: DMLabel = null_mut();
    let mut al: DMLabel = null_mut();
    petsc!(DMGetLabel(dm, walls.as_ptr(), &mut wl));
    petsc!(DMGetLabel(dm, areas.as_ptr(), &mut al));
    assert!(!wl.is_null() && !al.is_null(), "labels lost");
    let mut cells = std::vec::Vec::new();
    let mut cell_area = std::vec::Vec::new();
    for c in c0..c1 {
        let mut n = 0;
        let mut cl: *mut PetscInt = null_mut();
        petsc!(DMPlexGetTransitiveClosure(dm, c, true, &mut n, &mut cl));
        let pts = unsafe { std::slice::from_raw_parts(cl, 2 * n as usize) };
        let vs: std::vec::Vec<usize> = pts
            .iter()
            .step_by(2)
            .filter(|&&p| p >= v0 && p < v1)
            .map(|&p| (p - v0) as usize)
            .collect();
        petsc!(DMPlexRestoreTransitiveClosure(dm, c, true, &mut n, &mut cl));
        assert_eq!(vs.len(), 3);
        cells.push([vs[0], vs[1], vs[2]]);
        let mut val = 0;
        petsc!(DMLabelGetValue(al, c, &mut val));
        cell_area.push(val as i64);
    }
    let mut edges = std::vec::Vec::new();
    for e in e0..e1 {
        let mut cone: *const PetscInt = null();
        petsc!(DMPlexGetCone(dm, e, &mut cone));
        let ab = unsafe { [(*cone - v0) as usize, (*cone.add(1) - v0) as usize] };
        let mut val = 0;
        petsc!(DMLabelGetValue(wl, e, &mut val));
        edges.push((ab, val as i64));
    }
    Output {
        coords,
        cells,
        edges,
        cell_area,
    }
}

/// FNV-1a over the mesh in DMPlex order: coordinates (bits), triangles, edges and labels.
fn hash(out: &Output) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    let mut eat = |x: u64| {
        for b in x.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    };
    eat(out.coords.len() as u64);
    for p in &out.coords {
        eat(p[0].to_bits());
        eat(p[1].to_bits());
    }
    eat(out.cells.len() as u64);
    for (c, a) in out.cells.iter().zip(&out.cell_area) {
        for &v in c {
            eat(v as u64);
        }
        eat(*a as u64);
    }
    eat(out.edges.len() as u64);
    for (e, t) in &out.edges {
        eat(e[0] as u64);
        eat(e[1] as u64);
        eat(*t as u64);
    }
    h
}

fn write_vtu(path: &str, out: &Output) {
    let tagged: std::vec::Vec<_> = out.edges.iter().filter(|e| e.1 > 0).collect();
    let n = out.cells.len() + tagged.len();
    let mut s = String::new();
    writeln!(s, "<?xml version=\"1.0\"?>").unwrap();
    writeln!(
        s,
        "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">"
    )
    .unwrap();
    writeln!(
        s,
        "<UnstructuredGrid><Piece NumberOfPoints=\"{}\" NumberOfCells=\"{n}\">",
        out.coords.len()
    )
    .unwrap();
    s.push_str("<Points><DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n");
    for p in &out.coords {
        writeln!(s, "{:e} {:e} 0", p[0], p[1]).unwrap();
    }
    s.push_str("</DataArray></Points><Cells>\n<DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n");
    for c in &out.cells {
        writeln!(s, "{} {} {}", c[0], c[1], c[2]).unwrap();
    }
    for (e, _) in &tagged {
        writeln!(s, "{} {}", e[0], e[1]).unwrap();
    }
    s.push_str("</DataArray>\n<DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n");
    let mut off = 0;
    for _ in &out.cells {
        off += 3;
        writeln!(s, "{off}").unwrap();
    }
    for _ in &tagged {
        off += 2;
        writeln!(s, "{off}").unwrap();
    }
    s.push_str("</DataArray>\n<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n");
    for _ in &out.cells {
        s.push_str("5\n");
    }
    for _ in &tagged {
        s.push_str("3\n");
    }
    s.push_str("</DataArray></Cells>\n<CellData Scalars=\"area\">\n<DataArray type=\"Int32\" Name=\"area\" format=\"ascii\">\n");
    for a in &out.cell_area {
        writeln!(s, "{a}").unwrap();
    }
    for _ in &tagged {
        s.push_str("0\n");
    }
    s.push_str("</DataArray>\n<DataArray type=\"Int32\" Name=\"wall\" format=\"ascii\">\n");
    for _ in &out.cells {
        s.push_str("0\n");
    }
    for (_, t) in &tagged {
        writeln!(s, "{t}").unwrap();
    }
    s.push_str("</DataArray></CellData></Piece></UnstructuredGrid></VTKFile>\n");
    std::fs::write(path, s).unwrap_or_else(|e| panic!("cannot write {path}: {e}"));
}

/// Minimum angle (degrees) and SICN `4 sqrt(3) A / sum l^2` of a triangle.
fn quality(p: [P; 3]) -> (f64, f64) {
    let mut min_angle = f64::INFINITY;
    let mut l2 = 0.0;
    for i in 0..3 {
        let (a, b, c) = (p[i], p[(i + 1) % 3], p[(i + 2) % 3]);
        let (u, v) = (sub(b, a), sub(c, a));
        min_angle = min_angle.min(libm::atan2(cross(u, v).abs(), dot(u, v)).to_degrees());
        l2 += dot(u, u);
    }
    let area = 0.5 * cross(sub(p[1], p[0]), sub(p[2], p[0]));
    (min_angle, 4.0 * 3f64.sqrt() * area / l2)
}

/// Quality, per-tag and per-area checks of a mesh against the input geometry.
fn report(geo: &Geometry, out: &Output, split_counts: &[PetscInt]) {
    let (mut min_angle, mut min_sicn) = (f64::INFINITY, f64::INFINITY);
    let mut inverted = 0;
    for c in &out.cells {
        let (a, s) = quality([out.coords[c[0]], out.coords[c[1]], out.coords[c[2]]]);
        min_angle = min_angle.min(a);
        min_sicn = min_sicn.min(s.abs());
        if s <= 0.0 {
            inverted += 1;
        }
    }
    // Orientation: DMPlex keeps one orientation for all cells; report the count of the other.
    println!("min angle            {min_angle:.3} deg");
    println!("min SICN             {min_sicn:.4}");
    println!("clockwise cells      {inverted}");
    let (mut rmin, mut rmax, mut rbad) = (f64::INFINITY, 0.0f64, 0);
    for (e, _) in &out.edges {
        let (a, b) = (out.coords[e[0]], out.coords[e[1]]);
        let r = dist(a, b) / geo.h([(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0]);
        rmin = rmin.min(r);
        rmax = rmax.max(r);
        if !(0.4..=2.0).contains(&r) {
            rbad += 1;
            if rbad <= 5 {
                println!(
                    "  bad edge ({:.3}, {:.3})-({:.3}, {:.3}) l {:.4} h(mid) {:.4} ratio {r:.4}",
                    a[0],
                    a[1],
                    b[0],
                    b[1],
                    dist(a, b),
                    dist(a, b) / r
                );
            }
        }
    }
    println!("edge ratio           min {rmin:.4} max {rmax:.4} outside [0.4, 2.0] {rbad}");
    // Per-tag lengths against the input polylines.
    let input_len = geo.tag_lengths();
    let mut out_len = vec![0.0; geo.tags.len()];
    let mut out_cnt = vec![0usize; geo.tags.len()];
    let mut bad_tag = 0;
    for (e, t) in &out.edges {
        if *t > 0 {
            let t = (*t - 1) as usize;
            if t >= geo.tags.len() {
                bad_tag += 1;
                continue;
            }
            out_len[t] += dist(out.coords[e[0]], out.coords[e[1]]);
            out_cnt[t] += 1;
        }
    }
    let mut max_rel = 0.0f64;
    for (t, name) in geo.tags.iter().enumerate() {
        let rel = (out_len[t] - input_len[t]).abs() / input_len[t];
        max_rel = max_rel.max(rel);
        println!(
            "tag {name:<16} edges {:5} (split label {:5}) length {:.12} input {:.12} rel {rel:.2e}",
            out_cnt[t], split_counts[t], out_len[t], input_len[t]
        );
    }
    println!("tag length max rel   {max_rel:.2e}");
    println!("unknown tag values   {bad_tag}");
    // Every input segment: the output edges on it carry its tag and cover its length.
    let scale = 1e-9;
    let mut seg_bad_tag = 0;
    let mut seg_max_rel = 0.0f64;
    let mut on_any = vec![false; out.edges.len()];
    for c in &geo.chains {
        for (a, b) in c.segments() {
            let l = dist(a, b);
            let mut sum = 0.0;
            for (i, (e, t)) in out.edges.iter().enumerate() {
                let (p, q) = (out.coords[e[0]], out.coords[e[1]]);
                if seg_dist(p, a, b) < scale && seg_dist(q, a, b) < scale {
                    on_any[i] = true;
                    sum += dist(p, q);
                    if *t != c.tag as i64 + 1 {
                        seg_bad_tag += 1;
                    }
                }
            }
            seg_max_rel = seg_max_rel.max((sum - l).abs() / l);
        }
    }
    let stray = out
        .edges
        .iter()
        .zip(&on_any)
        .filter(|((_, t), on)| *t > 0 && !**on)
        .count();
    println!(
        "segments             sub-edges with another tag {seg_bad_tag}, tagged edges off the input {stray}, max rel length {seg_max_rel:.2e}"
    );
    // Every cell keeps its area tag (by its centroid against the input polygons).
    let mut area_bad = 0;
    for (c, &a) in out.cells.iter().zip(&out.cell_area) {
        let p = [out.coords[c[0]], out.coords[c[1]], out.coords[c[2]]];
        let centroid = [
            (p[0][0] + p[1][0] + p[2][0]) / 3.0,
            (p[0][1] + p[1][1] + p[2][1]) / 3.0,
        ];
        if geo.area_of(centroid).map(|x| x as i64 + 1) != Some(a) {
            area_bad += 1;
        }
    }
    let mut area_cnt = vec![0usize; geo.areas.len()];
    for &a in &out.cell_area {
        if a >= 1 && (a as usize) <= geo.areas.len() {
            area_cnt[a as usize - 1] += 1;
        }
    }
    for (i, name) in geo.areas.iter().enumerate() {
        println!("area {name:<15} cells {}", area_cnt[i]);
    }
    println!("cells with another area {area_bad}");
}

/// One `DMAdaptMetric` of `dm` (labels `walls`, `areas`) to the isotropic metric 1/h^2 of the
/// size field at its vertices.
fn adapt(dm: DM, geo: &Geometry, walls: &CStr, areas: &CStr) -> DM {
    let (mut v0, mut v1) = (0, 0);
    petsc!(DMPlexGetDepthStratum(dm, 0, &mut v0, &mut v1));
    let mut csec: PetscSection = null_mut();
    let mut cvec: Vec = null_mut();
    petsc!(DMGetCoordinateSection(dm, &mut csec));
    petsc!(DMGetCoordinatesLocal(dm, &mut cvec));
    petsc!(DMPlexMetricSetIsotropic(dm, true));
    let mut indicator: Vec = null_mut();
    petsc!(DMPlexMetricCreate(dm, 0, &mut indicator));
    let mut sec: PetscSection = null_mut();
    petsc!(DMGetLocalSection(dm, &mut sec));
    let mut arr: *mut PetscScalar = null_mut();
    let mut xy: *const PetscScalar = null();
    petsc!(VecGetArray(indicator, &mut arr));
    petsc!(VecGetArrayRead(cvec, &mut xy));
    for v in v0..v1 {
        let (mut off, mut coff) = (0, 0);
        petsc!(PetscSectionGetOffset(sec, v, &mut off));
        petsc!(PetscSectionGetOffset(csec, v, &mut coff));
        let p = unsafe { [*xy.add(coff as usize), *xy.add(coff as usize + 1)] };
        let h = geo.h(p);
        unsafe { *arr.add(off as usize) = 1.0 / (h * h) };
    }
    petsc!(VecRestoreArrayRead(cvec, &mut xy));
    petsc!(VecRestoreArray(indicator, &mut arr));
    let mut metric: Vec = null_mut();
    petsc!(DMPlexMetricCreateIsotropic(dm, 0, indicator, &mut metric));
    let mut wl: DMLabel = null_mut();
    let mut al: DMLabel = null_mut();
    petsc!(DMGetLabel(dm, walls.as_ptr(), &mut wl));
    petsc!(DMGetLabel(dm, areas.as_ptr(), &mut al));
    let mut adapted: DM = null_mut();
    petsc!(DMAdaptMetric(dm, metric, wl, al, &mut adapted));
    assert!(!adapted.is_null(), "DMAdaptMetric returned no mesh");
    petsc!(VecDestroy(&mut metric));
    petsc!(VecDestroy(&mut indicator));
    adapted
}

fn main() {
    let args: std::vec::Vec<CString> = std::env::args().map(|a| CString::new(a).unwrap()).collect();
    let mut argv: std::vec::Vec<*mut c_char> =
        args.iter().map(|a| a.as_ptr() as *mut c_char).collect();
    argv.push(null_mut());
    let mut argc = args.len() as c_int;
    let mut argv_ptr = argv.as_mut_ptr();
    petsc!(PetscInitialize(&mut argc, &mut argv_ptr, null(), null()));
    let comm = unsafe { PETSC_COMM_WORLD };
    default_option(c"-dm_adaptor", c"mmg");
    default_option(c"-dm_plex_metric_no_surf", c"1");

    let fixture = opt_string(c"-fixture").unwrap_or_else(|| "lobule".into());
    let geo = match fixture.as_str() {
        "lobule" => lobule(),
        "split" => split(),
        // The interior chain between two cells of the same area (both areas merged): does Mmg
        // keep its tag?
        "split1" => {
            let mut g = split();
            g.areas = &["tissue"];
            g.area_polys = vec![g.chains[0].pts.clone()];
            g
        }
        f => panic!("unknown fixture {f} (lobule, split, split1)"),
    };
    let refine_angle = opt_real(c"-refine_angle", 0.0);
    let refine_area = opt_real(c"-refine_area", 700.0);

    let t0 = Instant::now();
    let mut exclude = false;
    petsc!(PetscOptionsGetBool(
        null_mut(),
        null(),
        c"-refine_exclude".as_ptr(),
        &mut exclude,
        null_mut()
    ));
    let input = triangulate(&geo, refine_angle, refine_area, exclude);
    let t_cdt = t0.elapsed();

    // DMPlex with the labels.
    let t1 = Instant::now();
    let cells: std::vec::Vec<PetscInt> = input
        .cells
        .iter()
        .flat_map(|c| c.iter().map(|&v| v as PetscInt))
        .collect();
    let coords: std::vec::Vec<PetscReal> = input.coords.iter().flat_map(|p| [p[0], p[1]]).collect();
    let mut dm: DM = null_mut();
    petsc!(DMPlexCreateFromCellListPetsc(
        comm,
        2,
        input.cells.len() as PetscInt,
        input.coords.len() as PetscInt,
        3,
        true,
        cells.as_ptr(),
        2,
        coords.as_ptr(),
        &mut dm
    ));
    let (walls, areas) = (c"walls", c"areas");
    petsc!(DMCreateLabel(dm, walls.as_ptr()));
    petsc!(DMCreateLabel(dm, areas.as_ptr()));
    let mut wl: DMLabel = null_mut();
    let mut al: DMLabel = null_mut();
    petsc!(DMGetLabel(dm, walls.as_ptr(), &mut wl));
    petsc!(DMGetLabel(dm, areas.as_ptr(), &mut al));
    let (mut v0, mut v1) = (0, 0);
    petsc!(DMPlexGetDepthStratum(dm, 0, &mut v0, &mut v1));
    for &([a, b], tag) in &input.edges {
        let pts = [v0 + a as PetscInt, v0 + b as PetscInt];
        let mut n = 0;
        let mut covered: *const PetscInt = null();
        petsc!(DMPlexGetJoin(dm, 2, pts.as_ptr(), &mut n, &mut covered));
        assert_eq!(n, 1, "constraint edge not in the mesh");
        let e = unsafe { *covered };
        petsc!(DMPlexRestoreJoin(dm, 2, pts.as_ptr(), &mut n, &mut covered));
        petsc!(DMLabelSetValue(wl, e, tag as PetscInt + 1));
    }
    for (c, &a) in input.cell_area.iter().enumerate() {
        petsc!(DMLabelSetValue(al, c as PetscInt, a as PetscInt + 1));
    }
    let mut report_input = false;
    petsc!(PetscOptionsHasName(
        null_mut(),
        null(),
        c"-report_input".as_ptr(),
        &mut report_input
    ));
    if report_input {
        println!("-- input mesh (spade)");
        let inp = read_dm(dm, walls, areas);
        let counts: std::vec::Vec<PetscInt> = (0..geo.tags.len())
            .map(|t| inp.edges.iter().filter(|e| e.1 == t as i64 + 1).count() as PetscInt)
            .collect();
        report(&geo, &inp, &counts);
        if let Some(path) = opt_string(c"-vtu_input") {
            write_vtu(&path, &inp);
        }
        println!("-- output mesh (Mmg)");
    }
    let t_plex = t1.elapsed();

    // Adaptation passes: the metric 1/h^2 of the size field at the vertices of the current mesh
    // (Mmg interpolates it linearly on that mesh for its new vertices).
    let mut passes: PetscInt = 1;
    petsc!(PetscOptionsGetInt(
        null_mut(),
        null(),
        c"-passes".as_ptr(),
        &mut passes,
        null_mut()
    ));
    let t2 = Instant::now();
    let mut adapted: DM = null_mut();
    for pass in 0..passes {
        let cur = if pass == 0 { dm } else { adapted };
        let mut next = adapt(cur, &geo, walls, areas);
        if pass > 0 {
            petsc!(DMDestroy(&mut adapted));
        }
        std::mem::swap(&mut adapted, &mut next);
    }
    let t_adapt = t2.elapsed();

    // Labels split back: one label per tag (value 1) and per area.
    let t3 = Instant::now();
    let out = read_dm(adapted, walls, areas);
    let mut split_counts = std::vec::Vec::new();
    for (t, name) in geo.tags.iter().enumerate() {
        let cname = CString::new(*name).unwrap();
        petsc!(DMCreateLabel(adapted, cname.as_ptr()));
        let mut l: DMLabel = null_mut();
        petsc!(DMGetLabel(adapted, cname.as_ptr(), &mut l));
        let mut is: IS = null_mut();
        petsc!(DMGetStratumIS(
            adapted,
            walls.as_ptr(),
            t as PetscInt + 1,
            &mut is
        ));
        let mut n = 0;
        if !is.is_null() {
            petsc!(ISGetLocalSize(is, &mut n));
            let mut idx: *const PetscInt = null();
            petsc!(ISGetIndices(is, &mut idx));
            for i in 0..n as usize {
                petsc!(DMLabelSetValue(l, *idx.add(i), 1));
            }
            petsc!(ISRestoreIndices(is, &mut idx));
            petsc!(ISDestroy(&mut is));
        }
        let mut size = 0;
        petsc!(DMGetStratumSize(adapted, cname.as_ptr(), 1, &mut size));
        split_counts.push(size);
    }
    let t_split = t3.elapsed();
    let t_mesh = t0.elapsed();

    // Report.
    println!("fixture              {fixture}");
    println!("input vertices       {}", input.coords.len());
    println!("input triangles      {}", input.cells.len());
    println!("input tagged edges   {}", input.edges.len());
    println!("vertices             {}", out.coords.len());
    println!("triangles            {}", out.cells.len());
    println!("edges                {}", out.edges.len());
    report(&geo, &out, &split_counts);
    println!("hash                 {:016x}", hash(&out));
    println!(
        "wall time            cdt {:.1} ms, plex+metric {:.1} ms, adapt {:.1} ms, split labels {:.1} ms, total {:.1} ms",
        t_cdt.as_secs_f64() * 1e3,
        t_plex.as_secs_f64() * 1e3,
        t_adapt.as_secs_f64() * 1e3,
        t_split.as_secs_f64() * 1e3,
        t_mesh.as_secs_f64() * 1e3
    );
    if let Some(path) = opt_string(c"-vtu") {
        write_vtu(&path, &out);
        println!("vtu                  {path}");
    }

    petsc!(DMDestroy(&mut adapted));
    petsc!(DMDestroy(&mut dm));
    petsc!(PetscFinalize());
}
