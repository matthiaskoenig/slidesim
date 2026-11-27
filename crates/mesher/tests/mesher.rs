//! Tests of the mesher: tagged polygons to an adapted, tagged triangle mesh. nextest runs each
//! test in its own process, so every test which adapts initializes PETSc once.

use std::collections::HashMap;
use std::time::Instant;

use contracts::{AreaPolygon, EdgeChain, Polygons, SizeField};
use mesher::{Domain, MeshTag, MesherError, Size, TaggedMesh, generate, triangulate};
use petsc::Petsc;

type P = [f64; 2];

/// The wall: a circle of radius 20 um around (25, 50) um, polygonized at 2 um, closed (the
/// first point repeated).
fn hole(centre: P, radius: f64) -> Vec<P> {
    let n = (2.0 * std::f64::consts::PI * radius / 2.0).ceil() as usize;
    let mut ring: Vec<P> = (0..n)
        .map(|k| {
            let a = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
            [centre[0] + radius * a.cos(), centre[1] + radius * a.sin()]
        })
        .collect();
    ring.push(ring[0]);
    ring
}

fn area(tag: &str, rings: Vec<Vec<P>>) -> AreaPolygon {
    AreaPolygon {
        tag: tag.to_string(),
        value: 1,
        rings,
    }
}

fn edge(tag: &str, points: Vec<P>) -> EdgeChain {
    EdgeChain {
        tag: tag.to_string(),
        value: 1,
        points,
    }
}

/// The fixture scaled by `s`: a square of 100 s um split at x = 50 s into the areas `left` and
/// `right`, the split tagged `border`, a hole of radius 20 s at (25 s, 50 s) in `left`
/// (polygonized at 2 um) tagged `wall`, the outer sides `cut`.
fn scaled(s: f64) -> Polygons {
    let p = |x: f64, y: f64| [s * x, s * y];
    let wall = hole(p(25.0, 50.0), 20.0 * s);
    Polygons {
        areas: vec![
            area(
                "left",
                vec![
                    vec![p(0., 0.), p(50., 0.), p(50., 100.), p(0., 100.), p(0., 0.)],
                    wall.clone(),
                ],
            ),
            area(
                "right",
                vec![vec![
                    p(50., 0.),
                    p(100., 0.),
                    p(100., 100.),
                    p(50., 100.),
                    p(50., 0.),
                ]],
            ),
        ],
        edges: vec![
            edge("border", vec![p(50., 0.), p(50., 100.)]),
            edge(
                "cut",
                vec![
                    p(50., 0.),
                    p(100., 0.),
                    p(100., 100.),
                    p(50., 100.),
                    p(0., 100.),
                    p(0., 0.),
                    p(50., 0.),
                ],
            ),
            edge("wall", wall),
        ],
    }
}

fn fixture() -> Polygons {
    scaled(1.0)
}

fn size_field() -> SizeField {
    SizeField {
        min_um: 2.0,
        max_um: 10.0,
        dist_min_um: 4.0,
        dist_max_um: 40.0,
        walls: vec!["wall".to_string()],
    }
}

fn domain_and_size(p: &Polygons) -> (Domain, Size) {
    let domain = Domain::from_polygons(p).unwrap();
    let size = Size::new(&size_field(), &domain).unwrap();
    (domain, size)
}

fn mesh(p: &Polygons) -> (TaggedMesh, mesher::Quality) {
    let petsc = Petsc::init(&[]).unwrap();
    let (domain, size) = domain_and_size(p);
    generate(&petsc, &domain, &size).unwrap()
}

fn triangulate_err(p: &Polygons) -> MesherError {
    let (domain, size) = domain_and_size(p);
    triangulate(&domain, &size).unwrap_err()
}

fn dist(a: P, b: P) -> f64 {
    (b[0] - a[0]).hypot(b[1] - a[1])
}

fn polyline_length(points: &[P]) -> f64 {
    points.windows(2).map(|w| dist(w[0], w[1])).sum()
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

fn tag_index(mesh: &TaggedMesh, name: &str) -> u16 {
    mesh.tags.iter().position(|t| t.name == name).unwrap() as u16
}

fn triangle_area(mesh: &TaggedMesh, t: [u32; 3]) -> f64 {
    let [a, b, c] = t.map(|i| mesh.vertices[i as usize]);
    0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
}

/// The number of triangles of every edge `[a, b]` (`a < b`) of the mesh.
fn edge_triangles(mesh: &TaggedMesh) -> HashMap<[u32; 2], usize> {
    let mut count = HashMap::new();
    for t in &mesh.triangles {
        for i in 0..3 {
            let (a, b) = (t[i], t[(i + 1) % 3]);
            *count.entry([a.min(b), a.max(b)]).or_insert(0) += 1;
        }
    }
    count
}

/// The summed length of the edges of tag `name`.
fn tag_length(mesh: &TaggedMesh, name: &str) -> f64 {
    let tag = tag_index(mesh, name);
    mesh.edges
        .iter()
        .zip(&mesh.edge_tags)
        .filter(|&(_, &t)| t == tag)
        .map(|(e, _)| dist(mesh.vertices[e[0] as usize], mesh.vertices[e[1] as usize]))
        .sum()
}

fn assert_rel(value: f64, expected: f64, tol: f64, what: &str) {
    let rel = ((value - expected) / expected).abs();
    assert!(rel <= tol, "{what}: {value} vs {expected} (rel {rel:e})");
}

#[test]
fn tags_are_preserved() {
    let p = fixture();
    let (mesh, _) = mesh(&p);
    // Every boundary edge carries exactly one tag.
    let triangles = edge_triangles(&mesh);
    let mut tags_of: HashMap<[u32; 2], usize> = HashMap::new();
    for e in &mesh.edges {
        assert!(e[0] < e[1], "edge {e:?} not ordered");
        *tags_of.entry(*e).or_insert(0) += 1;
    }
    for (e, &n) in &triangles {
        if n == 1 {
            assert_eq!(tags_of.get(e).copied(), Some(1), "boundary edge {e:?}");
        }
    }
    // Per tag, the length of the input polylines.
    assert_rel(tag_length(&mesh, "border"), 100.0, 1e-12, "border");
    assert_rel(tag_length(&mesh, "cut"), 400.0, 1e-12, "cut");
    let wall = polyline_length(&p.edges[2].points);
    assert_rel(tag_length(&mesh, "wall"), wall, 1e-12, "wall");
    // Per area tag, the area of its polygon.
    let (left, right) = (tag_index(&mesh, "left"), tag_index(&mesh, "right"));
    let mut sums = [0.0; 2];
    for (t, &tag) in mesh.triangles.iter().zip(&mesh.triangle_tags) {
        let a = triangle_area(&mesh, *t);
        assert!(a > 0.0, "triangle {t:?} is not counterclockwise");
        if tag == left {
            sums[0] += a;
        } else if tag == right {
            sums[1] += a;
        } else {
            panic!("triangle {t:?} has the tag {tag}");
        }
    }
    let left_area = 5000.0 - ring_area(&p.areas[0].rings[1]);
    assert_rel(sums[0], left_area, 1e-12, "left");
    assert_rel(sums[1], 5000.0, 1e-12, "right");
}

#[test]
fn size_field_is_honoured() {
    let (_, q) = mesh(&fixture());
    let [lo, hi] = q.edge_ratio.unwrap();
    assert!(lo >= 0.4 && hi <= 2.0, "edge ratio {lo} to {hi}");
}

#[test]
fn quality_bound() {
    let (_, q) = mesh(&fixture());
    assert!(q.min_angle_deg >= 20.0, "min angle {}", q.min_angle_deg);
    assert!(q.min_sicn >= 0.4, "min SICN {}", q.min_sicn);
}

#[test]
fn deterministic() {
    let petsc = Petsc::init(&[]).unwrap();
    let (domain, size) = domain_and_size(&fixture());
    let (a, _) = generate(&petsc, &domain, &size).unwrap();
    let (b, _) = generate(&petsc, &domain, &size).unwrap();
    assert!(a == b, "two runs differ");
}

#[test]
fn untagged_boundary_segment_is_an_error() {
    let mut p = fixture();
    // Drop the segment (0, 0) to (50, 0) of `cut`.
    p.edges[1].points.pop();
    match triangulate_err(&p) {
        MesherError::UntaggedBoundary { at } => assert_eq!(at, [25.0, 0.0]),
        e => panic!("unexpected {e:?}"),
    }
}

#[test]
fn overlapping_areas_are_an_error() {
    let mut p = fixture();
    for q in &mut p.areas[1].rings[0] {
        q[0] -= 1.0;
    }
    match triangulate_err(&p) {
        MesherError::Overlap { at } => {
            assert!(
                at[0] > 49.0 && at[0] < 50.0 && at[1] > 0.0 && at[1] < 100.0,
                "{at:?}"
            )
        }
        e => panic!("unexpected {e:?}"),
    }
}

#[test]
fn edge_not_on_a_boundary_is_an_error() {
    let mut p = fixture();
    p.edges
        .push(edge("scratch", vec![[10.0, 10.0], [40.0, 20.0]]));
    match triangulate_err(&p) {
        MesherError::EdgeOffBoundary { tag, at } => {
            assert_eq!(tag, "scratch");
            assert_eq!(at, [25.0, 15.0]);
        }
        e => panic!("unexpected {e:?}"),
    }
}

#[test]
fn crossing_chains_and_rings_are_an_error() {
    // A chain crossing `border`.
    let mut p = fixture();
    p.edges
        .push(edge("scratch", vec![[40.0, 90.0], [60.0, 95.0]]));
    match triangulate_err(&p) {
        MesherError::SelfIntersection { at } => {
            assert!(dist(at, [50.0, 92.5]) <= 1e-12, "{at:?}")
        }
        e => panic!("unexpected {e:?}"),
    }
    // A ring crossing itself (a bow tie).
    let mut p = fixture();
    p.areas[1].rings[0] = vec![
        [50.0, 0.0],
        [100.0, 100.0],
        [100.0, 0.0],
        [50.0, 100.0],
        [50.0, 0.0],
    ];
    match triangulate_err(&p) {
        MesherError::SelfIntersection { at } => assert_eq!(at, [75.0, 50.0]),
        e => panic!("unexpected {e:?}"),
    }
}

#[test]
fn chain_splitting_one_area_keeps_its_tag() {
    // The fixture with `left` and `right` merged into one area, split by a kinked chain into two
    // regions of the same area tag.
    let mut p = fixture();
    let wall = p.areas[0].rings[1].clone();
    p.areas = vec![area(
        "tissue",
        vec![
            vec![
                [0.0, 0.0],
                [50.0, 0.0],
                [100.0, 0.0],
                [100.0, 100.0],
                [50.0, 100.0],
                [0.0, 100.0],
                [0.0, 0.0],
            ],
            wall,
        ],
    )];
    let chain = vec![[50.0, 0.0], [55.0, 50.0], [50.0, 100.0]];
    p.edges[0].points = chain.clone();
    let (mesh, q) = mesh(&p);
    let tissue = tag_index(&mesh, "tissue");
    assert!(mesh.triangle_tags.iter().all(|&t| t == tissue));
    assert_rel(
        tag_length(&mesh, "border"),
        polyline_length(&chain),
        1e-12,
        "border",
    );
    let border = tag_index(&mesh, "border");
    let triangles = edge_triangles(&mesh);
    let edges: Vec<_> = mesh
        .edges
        .iter()
        .zip(&mesh.edge_tags)
        .filter(|&(_, &t)| t == border)
        .map(|(e, _)| *e)
        .collect();
    assert!(edges.len() > 10, "{} border edges", edges.len());
    for e in edges {
        assert_eq!(triangles[&e], 2, "border edge {e:?}");
    }
    let total: f64 = mesh
        .triangles
        .iter()
        .map(|&t| triangle_area(&mesh, t))
        .sum();
    assert_rel(
        total,
        10000.0 - ring_area(&p.areas[0].rings[1]),
        1e-12,
        "area",
    );
    assert!(q.min_angle_deg >= 20.0, "min angle {}", q.min_angle_deg);
}

#[test]
fn dm_round_trip() {
    let petsc = Petsc::init(&[]).unwrap();
    let (domain, size) = domain_and_size(&fixture());
    let (mesh, _) = generate(&petsc, &domain, &size).unwrap();
    let dm = mesh.to_dm(&petsc.comm_self()).unwrap();
    assert!(TaggedMesh::from_dm(&dm).unwrap() == mesh);
}

#[test]
fn unknown_wall_tag_is_an_error() {
    let domain = Domain::from_polygons(&fixture()).unwrap();
    let mut spec = size_field();
    spec.walls.push("vessel".to_string());
    assert!(matches!(
        Size::new(&spec, &domain),
        Err(MesherError::SizeField(m)) if m.contains("vessel")
    ));
}

fn one_triangle(vertices: Vec<P>) -> TaggedMesh {
    TaggedMesh {
        tags: vec![MeshTag {
            name: "a".to_string(),
            value: 1,
            dim: 2,
        }],
        vertices,
        triangles: vec![[0, 1, 2]],
        triangle_tags: vec![0],
        edges: vec![],
        edge_tags: vec![],
    }
}

#[test]
fn quality_of_known_triangles() {
    let equilateral = one_triangle(vec![[0.0, 0.0], [1.0, 0.0], [0.5, 3f64.sqrt() / 2.0]]);
    let q = equilateral.quality(None);
    assert!((q.min_sicn - 1.0).abs() <= 1e-15, "{}", q.min_sicn);
    assert!((q.mean_sicn - 1.0).abs() <= 1e-15, "{}", q.mean_sicn);
    assert!(
        (q.min_angle_deg - 60.0).abs() <= 1e-12,
        "{}",
        q.min_angle_deg
    );
    assert_eq!((q.triangles, q.vertices), (1, 3));
    assert_eq!(q.edge_ratio, None);
    let right = one_triangle(vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
    let q = right.quality(None);
    assert!(
        (q.min_sicn - 0.8660254037844386).abs() <= 1e-15,
        "{}",
        q.min_sicn
    );
    assert!(
        (q.min_angle_deg - 45.0).abs() <= 1e-12,
        "{}",
        q.min_angle_deg
    );
}

/// Peak resident memory of this process in kB (`VmHWM`).
fn peak_memory_kb() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    line.split_whitespace().nth(1)?.parse().ok()
}

/// The fixture scaled to a mesh of 50k to 200k vertices: run with `--run-ignored only`.
#[test]
#[ignore = "large: about 100k vertices"]
fn scale() {
    let petsc = Petsc::init(&[]).unwrap();
    let p = scaled(32.0);
    let (domain, size) = domain_and_size(&p);
    let start = Instant::now();
    let input = triangulate(&domain, &size).unwrap();
    let cdt = start.elapsed();
    let adapted = mesher::adapt(&petsc, &input, &size).unwrap();
    let total = start.elapsed();
    let q = adapted.quality(Some(&size));
    println!(
        "scale: input {} vertices {} triangles; output {} vertices {} triangles; \
         triangulate {:.3} s, total {:.3} s; peak memory {:?} kB; quality {:?}",
        input.vertices.len(),
        input.triangles.len(),
        q.vertices,
        q.triangles,
        cdt.as_secs_f64(),
        total.as_secs_f64(),
        peak_memory_kb(),
        (q.min_angle_deg, q.min_sicn, q.mean_sicn, q.edge_ratio),
    );
    assert!(
        (50_000..=200_000).contains(&q.vertices),
        "{} vertices",
        q.vertices
    );
    assert!(q.min_angle_deg >= 20.0);
    let [lo, hi] = q.edge_ratio.unwrap();
    assert!(lo >= 0.4 && hi <= 2.0, "edge ratio {lo} to {hi}");
    assert_rel(tag_length(&adapted, "cut"), 400.0 * 32.0, 1e-12, "cut");
}

#[test]
fn closed_chain_inside_an_area_is_kept() {
    // An island: a closed chain inside `right`, both its sides `right`.
    let mut p = fixture();
    let island = vec![
        [60.0, 40.0],
        [90.0, 40.0],
        [90.0, 60.0],
        [60.0, 60.0],
        [60.0, 40.0],
    ];
    p.edges.push(edge("island", island.clone()));
    let (mesh, q) = mesh(&p);
    assert_rel(
        tag_length(&mesh, "island"),
        polyline_length(&island),
        1e-12,
        "island",
    );
    let (right, tag) = (tag_index(&mesh, "right"), tag_index(&mesh, "island"));
    let inside: f64 = mesh
        .triangles
        .iter()
        .zip(&mesh.triangle_tags)
        .filter(|&(t, _)| {
            t.iter().all(|&v| {
                let [x, y] = mesh.vertices[v as usize];
                (60.0..=90.0).contains(&x) && (40.0..=60.0).contains(&y)
            })
        })
        .map(|(&t, &a)| {
            assert_eq!(a, right);
            triangle_area(&mesh, t)
        })
        .sum();
    assert_rel(inside, 600.0, 1e-12, "island area");
    let triangles = edge_triangles(&mesh);
    for (e, _) in mesh
        .edges
        .iter()
        .zip(&mesh.edge_tags)
        .filter(|&(_, &t)| t == tag)
    {
        assert_eq!(triangles[e], 2, "island edge {e:?}");
    }
    assert!(q.min_angle_deg >= 20.0, "min angle {}", q.min_angle_deg);
}

#[test]
fn steep_size_field_is_meshed() {
    // The size jumps from 2 to 10 um within 0.01 um: sub-edges far longer than the size next to
    // them, which no refinement of the background shortens. The quality is not bounded then, but
    // the meshing ends with the tags and areas intact.
    let petsc = Petsc::init(&[]).unwrap();
    let p = fixture();
    let domain = Domain::from_polygons(&p).unwrap();
    let spec = SizeField {
        dist_max_um: 4.01,
        ..size_field()
    };
    let size = Size::new(&spec, &domain).unwrap();
    let (mesh, q) = generate(&petsc, &domain, &size).unwrap();
    assert_rel(tag_length(&mesh, "cut"), 400.0, 1e-12, "cut");
    let wall = polyline_length(&p.edges[2].points);
    assert_rel(tag_length(&mesh, "wall"), wall, 1e-12, "wall");
    let total: f64 = mesh
        .triangles
        .iter()
        .map(|&t| triangle_area(&mesh, t))
        .sum();
    assert_rel(
        total,
        10000.0 - ring_area(&p.areas[0].rings[1]),
        1e-12,
        "area",
    );
    assert!(q.min_sicn > 0.0, "min SICN {}", q.min_sicn);
}

#[test]
fn untagged_interface_between_areas_is_kept() {
    // Without `border`, the interface of `left` and `right` has no tag: it needs none, and stays.
    let mut p = fixture();
    p.edges.remove(0);
    let (mesh, q) = mesh(&p);
    let (left, right) = (tag_index(&mesh, "left"), tag_index(&mesh, "right"));
    let mut sums = [0.0; 2];
    for (&t, &tag) in mesh.triangles.iter().zip(&mesh.triangle_tags) {
        let a = triangle_area(&mesh, t);
        let x = t.map(|v| mesh.vertices[v as usize][0]);
        if tag == left {
            assert!(x.iter().all(|&x| x <= 50.0), "{t:?}");
            sums[0] += a;
        } else {
            assert_eq!(tag, right);
            assert!(x.iter().all(|&x| x >= 50.0), "{t:?}");
            sums[1] += a;
        }
    }
    assert_rel(
        sums[0],
        5000.0 - ring_area(&p.areas[0].rings[1]),
        1e-12,
        "left",
    );
    assert_rel(sums[1], 5000.0, 1e-12, "right");
    assert!(q.min_angle_deg >= 20.0, "min angle {}", q.min_angle_deg);
}
