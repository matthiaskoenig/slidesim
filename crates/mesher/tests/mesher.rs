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

/// FNV-1a over the mesh: coordinates (bits), triangles, tags, edges.
fn hash(mesh: &TaggedMesh) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    let mut eat = |x: u64| {
        for b in x.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x100000001b3);
        }
    };
    for v in &mesh.vertices {
        eat(v[0].to_bits());
        eat(v[1].to_bits());
    }
    for (t, &a) in mesh.triangles.iter().zip(&mesh.triangle_tags) {
        t.iter().for_each(|&v| eat(v.into()));
        eat(a.into());
    }
    for (e, &a) in mesh.edges.iter().zip(&mesh.edge_tags) {
        e.iter().for_each(|&v| eat(v.into()));
        eat(a.into());
    }
    for t in &mesh.tags {
        t.name.bytes().for_each(|b| eat(b.into()));
        eat(t.value as u64);
        eat(t.dim.into());
    }
    h
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

/// The quality bounds the mesher guarantees for resolvable inputs.
fn assert_quality(q: &mesher::Quality) {
    assert!(q.min_angle_deg >= 20.0, "min angle {}", q.min_angle_deg);
    assert!(q.min_sicn >= 0.4, "min SICN {}", q.min_sicn);
    let [lo, hi] = q.edge_ratio.unwrap();
    assert!(lo >= 0.4 && hi <= 2.0, "edge ratio {lo} to {hi}");
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
    assert_quality(&q);
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
    let q = equilateral.quality(None).unwrap();
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
    let q = right.quality(None).unwrap();
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
    let q = adapted.quality(Some(&size)).unwrap();
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
    // An island: a closed chain inside `right`, both its sides `right`, 10 um from the cut (a
    // gap as wide as the size there).
    let mut p = fixture();
    let island = vec![
        [60.0, 10.0],
        [90.0, 10.0],
        [90.0, 30.0],
        [60.0, 30.0],
        [60.0, 10.0],
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
                (60.0..=90.0).contains(&x) && (10.0..=30.0).contains(&y)
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

#[test]
fn narrow_neck_is_resolved() {
    // One area: two 40 um squares joined by a neck 3 um wide and 20 um long, all of it `cut`; no
    // walls, so the size field alone would be 10 um everywhere.
    let ring = vec![
        [0.0, 0.0],
        [40.0, 0.0],
        [40.0, 18.5],
        [60.0, 18.5],
        [60.0, 0.0],
        [100.0, 0.0],
        [100.0, 40.0],
        [60.0, 40.0],
        [60.0, 21.5],
        [40.0, 21.5],
        [40.0, 40.0],
        [0.0, 40.0],
        [0.0, 0.0],
    ];
    let p = Polygons {
        areas: vec![area("tissue", vec![ring.clone()])],
        edges: vec![edge("cut", ring.clone())],
    };
    let petsc = Petsc::init(&[]).unwrap();
    let domain = Domain::from_polygons(&p).unwrap();
    let spec = SizeField {
        walls: vec![],
        ..size_field()
    };
    let size = Size::new(&spec, &domain).unwrap();
    assert!(
        size.at([50.0, 20.0]) < 3.0,
        "size {} in the neck",
        size.at([50.0, 20.0])
    );
    let (mesh, q) = generate(&petsc, &domain, &size).unwrap();
    assert_rel(
        tag_length(&mesh, "cut"),
        polyline_length(&ring),
        1e-12,
        "cut",
    );
    let total: f64 = mesh
        .triangles
        .iter()
        .map(|&t| triangle_area(&mesh, t))
        .sum();
    assert_rel(total, 2.0 * 1600.0 + 60.0, 1e-12, "area");
    assert_quality(&q);
}

#[test]
fn fixture_mesh_is_stable() {
    // The fixture's mesh, bit for bit, on this target: a change of the mesher, spade, Mmg or the
    // PETSc build shows here.
    let (mesh, _) = mesh(&fixture());
    assert_eq!(hash(&mesh), FIXTURE_HASH, "{:#018x}", hash(&mesh));
}

/// The hash of the fixture's mesh (x86_64 Linux, the native PETSc builds).
const FIXTURE_HASH: u64 = 0xfbe8d8772caefddc;

#[test]
fn perturbed_shared_border_is_an_error() {
    // A border shared by two areas which does not match exactly: a gap or an overlap of 1e-9 um.
    for (dx, gap) in [(1e-9, true), (-1e-9, false)] {
        let mut p = fixture();
        for q in &mut p.areas[1].rings[0] {
            if q[0] == 50.0 {
                q[0] += dx;
            }
        }
        let e = triangulate_err(&p);
        match &e {
            MesherError::UntaggedBoundary { .. } if gap => {}
            MesherError::Overlap { .. } if !gap => {}
            e => panic!("dx {dx}: unexpected {e:?}"),
        }
        assert!(e.to_string().contains("match exactly"), "{e}");
    }
}

#[test]
fn tags_are_sorted_and_round_trip() {
    // One tag name with two values (the second polygon's first): the tags sorted by name and
    // value, and the DMPlex round trip exact.
    let mut p = fixture();
    p.areas[0].value = 2;
    p.areas[1].tag = "left".to_string();
    p.areas[1].value = 1;
    let domain = Domain::from_polygons(&p).unwrap();
    let names: Vec<(&str, i32)> = domain
        .tags
        .iter()
        .map(|t| (t.name.as_str(), t.value))
        .collect();
    assert_eq!(
        names,
        [
            ("border", 1),
            ("cut", 1),
            ("left", 1),
            ("left", 2),
            ("wall", 1)
        ]
    );
    let petsc = Petsc::init(&[]).unwrap();
    let size = Size::new(&size_field(), &domain).unwrap();
    let (mesh, _) = generate(&petsc, &domain, &size).unwrap();
    let dm = mesh.to_dm(&petsc.comm_self()).unwrap();
    assert!(TaggedMesh::from_dm(&dm).unwrap() == mesh);
}

#[test]
fn reserved_and_oversized_tags_are_an_error() {
    let mut p = fixture();
    p.areas[0].tag = "depth".to_string();
    let e = Domain::from_polygons(&p).unwrap_err();
    assert!(
        matches!(&e, MesherError::InvalidTag(m) if m.contains("depth") && m.contains("reserved")),
        "{e:?}"
    );
    let mut p = fixture();
    p.edges[0].value = u32::MAX;
    let e = Domain::from_polygons(&p).unwrap_err();
    assert!(
        matches!(&e, MesherError::InvalidTag(m) if m.contains("border")),
        "{e:?}"
    );
    // A `TaggedMesh` read from a file is checked too.
    let petsc = Petsc::init(&[]).unwrap();
    let mut mesh = one_triangle(vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
    mesh.tags[0].name = "celltype".to_string();
    let e = mesh.to_dm(&petsc.comm_self()).unwrap_err();
    assert!(
        e.message.contains("celltype") && e.message.contains("reserved"),
        "{e}"
    );
}

#[test]
fn malformed_meshes_are_an_error() {
    let petsc = Petsc::init(&[]).unwrap();
    let good = one_triangle(vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
    let mut bad = Vec::new();
    let mut m = good.clone();
    m.triangles[0][2] = 7;
    bad.push(m);
    let mut m = good.clone();
    m.triangle_tags.clear();
    bad.push(m);
    let mut m = good.clone();
    m.triangle_tags[0] = 3;
    bad.push(m);
    let mut m = good.clone();
    m.tags.push(MeshTag {
        name: "e".to_string(),
        value: 1,
        dim: 1,
    });
    m.edges.push([0, 1]);
    bad.push(m.clone());
    m.edge_tags.push(1);
    m.edges[0] = [1, 0];
    bad.push(m.clone());
    m.vertices.push([5.0, 5.0]);
    m.edges[0] = [0, 3];
    bad.push(m);
    for m in &bad {
        assert!(
            matches!(m.validate(), Err(MesherError::InvalidMesh(_))),
            "{m:?}"
        );
        assert!(m.to_dm(&petsc.comm_self()).is_err(), "{m:?}");
    }
    for m in &bad[..3] {
        assert!(m.quality(None).is_err(), "{m:?}");
    }
    assert!(good.validate().is_ok());
}

/// The tests run again on 2 processes by `parallel_on_2_processes`.
const PARALLEL_TESTS: [&str; 2] = [
    "to_dm_has_every_label_on_every_process",
    "to_dm_error_on_process_0_reaches_every_process",
];

#[test]
fn parallel_on_2_processes() {
    test_support::parallel(2, &PARALLEL_TESTS);
}

#[test]
fn to_dm_has_every_label_on_every_process() {
    // The mesher runs on process 0; the others pass an empty mesh.
    let petsc = Petsc::init(&[]).unwrap();
    let comm = petsc.comm();
    assert_eq!(comm.size(), test_support::ranks());
    let (domain, size) = domain_and_size(&fixture());
    let mesh = if comm.rank() == 0 {
        generate(&petsc, &domain, &size).unwrap().0
    } else {
        TaggedMesh {
            tags: vec![],
            vertices: vec![],
            triangles: vec![],
            triangle_tags: vec![],
            edges: vec![],
            edge_tags: vec![],
        }
    };
    let dm = mesh.to_dm(&comm).unwrap();
    for tag in &domain.tags {
        assert!(dm.has_label(&tag.name).unwrap(), "{}", tag.name);
        assert_eq!(
            dm.label_values_global(&tag.name).unwrap(),
            [1],
            "{}",
            tag.name
        );
    }
}

#[test]
fn to_dm_error_on_process_0_reaches_every_process() {
    let petsc = Petsc::init(&[]).unwrap();
    let comm = petsc.comm();
    let mut mesh = one_triangle(vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
    if comm.rank() == 0 {
        mesh.triangles[0][1] = 9;
    }
    let e = mesh.to_dm(&comm).unwrap_err();
    assert!(e.message.contains("vertex"), "{e}");
}
