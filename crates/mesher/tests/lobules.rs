//! Tests of the parametric lobules: their geometry, their errors, the relative position between
//! the walls, and their mesh. nextest runs each test in its own process, so the test which
//! meshes initializes PETSc once.

use std::time::Instant;

use contracts::{HoleSpec, LobuleSize, LobuleSpec, ZonationSpec};
use mesher::{Domain, Lobules, MesherError, generate, triangulate};
use petsc::Petsc;

type P = [f64; 2];

/// `R` 300, centre 30, corner 25, wall 5, interior 20; the tags `area`, `centre`, `corner` and
/// `outer`.
fn spec(rings: u32, jitter: f64, seed: u64) -> LobuleSpec {
    LobuleSpec {
        lobule_radius_um: 300.0,
        rings,
        jitter,
        seed,
        centre: HoleSpec {
            radius_um: 30.0,
            tag: "centre".to_string(),
        },
        corner: HoleSpec {
            radius_um: 25.0,
            tag: "corner".to_string(),
        },
        outer_tag: "outer".to_string(),
        area_tag: "area".to_string(),
        size: LobuleSize {
            wall_um: 5.0,
            interior_um: 20.0,
        },
        zonation: ZonationSpec {
            map: "zonation".to_string(),
            um_per_px: 2.0,
        },
    }
}

fn lobules(rings: u32, jitter: f64, seed: u64) -> Lobules {
    Lobules::new(&spec(rings, jitter, seed)).unwrap()
}

fn dist(a: P, b: P) -> f64 {
    (b[0] - a[0]).hypot(b[1] - a[1])
}

/// The shoelace area of a ring (closed or not), unsigned.
fn ring_area(ring: &[P]) -> f64 {
    let n = ring.len();
    let twice: f64 = (0..n)
        .map(|i| {
            let (a, b) = (ring[i], ring[(i + 1) % n]);
            a[0] * b[1] - a[1] * b[0]
        })
        .sum();
    0.5 * twice.abs()
}

/// The area of the tissue: the outer rings less the holes.
fn tissue_area(domain: &Domain) -> f64 {
    domain
        .areas
        .iter()
        .flat_map(|(_, rings)| {
            rings
                .iter()
                .enumerate()
                .map(|(i, r)| if i == 0 { ring_area(r) } else { -ring_area(r) })
        })
        .sum()
}

/// The chains of the edge tag `name`.
fn chains<'a>(domain: &'a Domain, name: &str) -> Vec<&'a [P]> {
    let tag = domain
        .tags
        .iter()
        .position(|t| t.name == name && t.dim == 1)
        .unwrap_or_else(|| panic!("no edge tag {name:?}"));
    domain
        .edges
        .iter()
        .filter(|(t, _)| *t == tag)
        .map(|(_, points)| points.as_slice())
        .collect()
}

fn wall_length(domain: &Domain, name: &str) -> f64 {
    chains(domain, name)
        .iter()
        .flat_map(|points| points.windows(2).map(|w| dist(w[0], w[1])))
        .sum()
}

fn assert_relative(what: &str, value: f64, expected: f64, tolerance: f64) {
    let error = (value - expected).abs() / expected;
    assert!(
        error <= tolerance,
        "{what}: {value}, expected {expected} within {tolerance} relative (off by {error})"
    );
}

#[test]
fn single_lobule_area() {
    let l = lobules(0, 0.0, 0);
    assert_relative("area", tissue_area(l.domain()), 227072.43481658038, 1e-3);
}

#[test]
fn single_lobule_wall_lengths() {
    let l = lobules(0, 0.0, 0);
    let d = l.domain();
    assert_relative("centre", wall_length(d, "centre"), 188.49555921538757, 5e-3);
    assert_relative("corner", wall_length(d, "corner"), 314.1592653589793, 5e-3);
    assert_relative("outer", wall_length(d, "outer"), 1500.0, 5e-3);
    assert_eq!(l.veins().len(), 1);
    assert_eq!(l.corners().len(), 6);
}

#[test]
fn ring_1_geometry() {
    let l = lobules(1, 0.0, 0);
    let d = l.domain();
    assert_eq!(l.veins().len(), 7);
    assert_eq!(l.corners().len(), 24);
    assert_relative("area", tissue_area(d), 1589507.0437160626, 1e-3);
    assert_relative("corner", wall_length(d, "corner"), 2199.114857512855, 5e-3);
    assert_relative("centre", wall_length(d, "centre"), 1319.4689145077132, 5e-3);
    assert_relative("outer", wall_length(d, "outer"), 4500.0, 5e-3);
}

#[test]
fn jitter_is_seeded() {
    assert_eq!(lobules(1, 0.1, 1).domain(), lobules(1, 0.1, 1).domain());
    assert_ne!(lobules(1, 0.1, 1).domain(), lobules(1, 0.1, 2).domain());
    assert_ne!(lobules(1, 0.1, 1).veins(), lobules(1, 0.1, 2).veins());
    assert_eq!(lobules(1, 0.0, 1).domain(), lobules(1, 0.0, 2).domain());
    assert_eq!(lobules(1, 0.0, 1).domain(), lobules(1, 0.0, 0).domain());
}

#[test]
fn holes_overlap_is_an_error() {
    // Built directly, not through the contract's checks: adjacent corners are 300 um apart.
    let mut s = spec(0, 0.0, 0);
    s.corner.radius_um = 160.0;
    match Lobules::new(&s) {
        Err(MesherError::HolesOverlap { a, b }) => {
            assert_eq!((a.as_str(), b.as_str()), ("corner", "corner"));
        }
        other => panic!("expected HolesOverlap, got {other:?}"),
    }
}

#[test]
fn hole_reaching_the_border_is_an_error() {
    // The inradius of the cell is 259.81 um; the corners stay 300 um from the vein.
    let mut s = spec(0, 0.0, 0);
    s.centre.radius_um = 259.9;
    s.corner.radius_um = 1.0;
    match Lobules::new(&s) {
        Err(MesherError::HolesOverlap { a, b }) => {
            assert_eq!((a.as_str(), b.as_str()), ("centre", "outer"));
        }
        other => panic!("expected HolesOverlap, got {other:?}"),
    }
    s.centre.radius_um = 259.7;
    assert!(Lobules::new(&s).is_ok());
}

#[test]
fn position_on_walls() {
    for (rings, jitter) in [(0, 0.0), (1, 0.0), (1, 0.25)] {
        let l = lobules(rings, jitter, 3);
        let d = l.domain();
        for points in chains(d, "corner") {
            for &x in points {
                let p = l.position(x);
                assert!(p.abs() <= 1e-12, "position {p} at the corner wall {x:?}");
            }
        }
        for points in chains(d, "centre") {
            for &x in points {
                let p = l.position(x);
                assert!(
                    (p - 1.0).abs() <= 1e-12,
                    "position {p} at the centre wall {x:?}"
                );
            }
        }
    }
}

#[test]
fn position_is_monotone() {
    let l = lobules(0, 0.0, 0);
    let vein = l.veins()[0];
    let corner = l.corners()[0];
    let length = dist(corner, vein);
    let direction = [
        (vein[0] - corner[0]) / length,
        (vein[1] - corner[1]) / length,
    ];
    // 100 samples strictly between the corner wall (25 um from the corner) and the centre wall
    // (30 um from the vein).
    let (start, end) = (25.0, length - 30.0);
    let positions: Vec<f64> = (1..=100)
        .map(|k| {
            let t = start + (end - start) * k as f64 / 101.0;
            l.position([corner[0] + t * direction[0], corner[1] + t * direction[1]])
        })
        .collect();
    assert!(positions[0] > 0.0 && positions[99] < 1.0, "{positions:?}");
    for w in positions.windows(2) {
        assert!(w[0] < w[1], "not increasing: {} then {}", w[0], w[1]);
    }
    // Clamped inside the holes.
    assert_eq!(l.position(corner), 0.0);
    assert_eq!(l.position(vein), 1.0);
}

#[test]
fn bounds_hold_the_domain() {
    let l = lobules(1, 0.25, 7);
    let [lo, hi] = l.bounds();
    for (_, rings) in &l.domain().areas {
        for x in rings.iter().flatten() {
            assert!((0..2).all(|i| lo[i] <= x[i] && x[i] <= hi[i]), "{x:?}");
        }
    }
    // The outer ring of seven hexagons of circumradius 300 spans about 1300 x 1500 um.
    assert!(
        hi[0] - lo[0] > 1000.0 && hi[1] - lo[1] > 1000.0,
        "{lo:?} {hi:?}"
    );
}

#[test]
fn invalid_specs_are_errors() {
    let mut s = spec(0, 0.0, 0);
    s.jitter = 0.3;
    assert!(matches!(
        Lobules::new(&s),
        Err(MesherError::InvalidLobules(_))
    ));
    let mut s = spec(0, 0.0, 0);
    s.lobule_radius_um = f64::NAN;
    assert!(matches!(
        Lobules::new(&s),
        Err(MesherError::InvalidLobules(_))
    ));
    let mut s = spec(0, 0.0, 0);
    s.outer_tag = "corner".to_string();
    assert!(matches!(
        Lobules::new(&s),
        Err(MesherError::InvalidLobules(_))
    ));
    let mut s = spec(0, 0.0, 0);
    s.area_tag = "depth".to_string();
    assert!(matches!(Lobules::new(&s), Err(MesherError::InvalidTag(_))));
    let mut s = spec(0, 0.0, 0);
    s.size.wall_um = 0.0;
    assert!(matches!(Lobules::new(&s), Err(MesherError::SizeField(_))));
}

/// Every seed of the largest jitter gives lobules, for the largest holes the contract allows.
#[test]
fn largest_jitter_is_robust() {
    // (1 - 2 jitter) R / 2 = 75: centre + corner and 2 corner just below it.
    for (centre, corner) in [(37.4, 37.4), (73.9, 1.0), (1.0, 37.4)] {
        for seed in 0..200 {
            let mut s = spec(2, 0.25, seed);
            s.centre.radius_um = centre;
            s.corner.radius_um = corner;
            let l = Lobules::new(&s)
                .unwrap_or_else(|e| panic!("centre {centre} corner {corner} seed {seed}: {e}"));
            assert_eq!(l.veins().len(), 19);
            assert_eq!(l.corners().len(), 54);
        }
    }
}

/// The domains of the largest jitter triangulate: the shared vertices of the rings and chains
/// match, nothing overlaps.
#[test]
fn jittered_lobules_triangulate() {
    for seed in 0..3 {
        let l = lobules(1, 0.25, seed);
        triangulate(l.domain(), &l.size()).unwrap_or_else(|e| panic!("seed {seed}: {e}"));
    }
}

#[test]
fn lobule_meshes() {
    let petsc = Petsc::init(&[]).unwrap();
    let run = |rings: u32, jitter: f64| {
        let start = Instant::now();
        let l = lobules(rings, jitter, 1);
        let geometry = start.elapsed();
        let (mesh, quality) = generate(&petsc, l.domain(), &l.size()).unwrap();
        eprintln!(
            "rings {rings} jitter {jitter}: geometry {geometry:?}, total {:?}, {quality:?}",
            start.elapsed()
        );
        (mesh, quality)
    };
    let (mesh, quality) = run(0, 0.0);
    let mut names: Vec<&str> = mesh.tags.iter().map(|t| t.name.as_str()).collect();
    names.dedup();
    assert_eq!(names, ["area", "centre", "corner", "outer"]);
    assert!(quality.min_angle_deg >= 20.0, "{quality:?}");
    let [low, high] = quality.edge_ratio.unwrap();
    assert!(low >= 0.4 && high <= 2.0, "{quality:?}");
    let (again, _) = run(0, 0.0);
    assert_eq!(mesh, again);
    // Larger and jittered: measured (Mmg gives a smallest angle a little below 20 degrees at
    // some corner walls), with regression bounds.
    for (rings, jitter) in [(1, 0.0), (1, 0.25)] {
        let (_, quality) = run(rings, jitter);
        let [low, high] = quality.edge_ratio.unwrap();
        assert!(quality.min_angle_deg >= 18.0, "{quality:?}");
        assert!(low >= 0.4 && high <= 2.0, "{quality:?}");
    }
}
