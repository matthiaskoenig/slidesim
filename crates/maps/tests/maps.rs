use std::path::PathBuf;

use contracts::MapSpec;
use maps::{Map, MapError, Maps, write_test_map};

const FULL: [[f64; 2]; 2] = [[-1e9, -1e9], [1e9, 1e9]];

fn fixture(name: &str) -> MapSpec {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/maps");
    let json = std::fs::read_to_string(dir.join(format!("{name}.json"))).unwrap();
    let mut spec: MapSpec = serde_json::from_str(&json).unwrap();
    spec.path = dir.join(&spec.path);
    spec
}

fn f(x: [f64; 2]) -> f64 {
    2.0 + 0.5 * x[0] - 0.25 * x[1]
}

/// Deterministic points in [lo, hi] without a rand dependency.
fn points(n: usize, lo: [f64; 2], hi: [f64; 2]) -> Vec<[f64; 2]> {
    let mut s = 12345u64;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            [
                lo[0] + next() * (hi[0] - lo[0]),
                lo[1] + next() * (hi[1] - lo[1]),
            ]
        })
        .collect()
}

// 40 x 60 pixels, 2 um, origin [10, 20]: x in [10, 128], y in [20, 98].
const LO: [f64; 2] = [10.0, 20.0];
const HI: [f64; 2] = [128.0, 98.0];

#[test]
fn bilinear_is_exact_for_linear() {
    let map = Map::read(&fixture("linear"), FULL).unwrap();
    for x in points(50, LO, HI) {
        let v = map.sample(x).unwrap();
        assert!((v - f(x)).abs() < 1e-5, "{x:?}: {v} vs {}", f(x));
    }
}

#[test]
fn gradient_is_exact_for_linear() {
    let map = Map::read(&fixture("linear"), FULL).unwrap();
    let mut pts = points(50, LO, HI);
    // the border pixel rows and columns (one-sided differences)
    pts.extend(points(10, LO, [HI[0], LO[1]]));
    pts.extend(points(10, [LO[0], HI[1]], HI));
    pts.extend(points(10, LO, [LO[0], HI[1]]));
    pts.extend(points(10, [HI[0], LO[1]], HI));
    for x in pts {
        let g = map.gradient(x).unwrap();
        assert!(
            (g[0] - 0.5).abs() < 1e-5 && (g[1] + 0.25).abs() < 1e-5,
            "{x:?}: {g:?}"
        );
    }
}

#[test]
fn outside_is_error() {
    let map = Map::read(&fixture("linear"), FULL).unwrap();
    let err = map.sample([9.0, 30.0]).unwrap_err();
    assert!(matches!(err, MapError::Outside { .. }));
    let msg = err.to_string();
    assert!(
        msg.contains("linear") && msg.contains('9') && msg.contains("30"),
        "{msg}"
    );
    assert!(matches!(
        map.gradient([50.0, 99.0]),
        Err(MapError::Outside { .. })
    ));
}

#[test]
fn nan_and_no_value_are_errors() {
    let map = Map::read(&fixture("no_value"), FULL).unwrap();
    // NaN block at rows 10-12, cols 30-32: x = 70..74, y = 40..44
    for x in [[71.0, 41.0], [69.0, 39.0], [75.0, 45.0], [72.0, 42.0]] {
        assert!(
            matches!(map.sample(x), Err(MapError::NoValue { .. })),
            "{x:?}"
        );
    }
    // -1 at row 0, col 0: x = 10, y = 20
    assert!(matches!(
        map.sample([10.5, 20.5]),
        Err(MapError::NoValue { .. })
    ));
    assert!(matches!(
        map.gradient([10.5, 20.5]),
        Err(MapError::NoValue { .. })
    ));
    // away from the holes the map is the linear one
    assert!((map.sample([100.0, 80.0]).unwrap() - f([100.0, 80.0])).abs() < 1e-5);
}

#[test]
fn window_reads_only_needed_chunks() {
    let spec = fixture("linear");
    let full = Map::read(&spec, FULL).unwrap();
    assert_eq!(full.window_size(), [40, 60]);
    let bounds = [[60.0, 50.0], [65.0, 55.0]];
    let win = Map::read(&spec, bounds).unwrap();
    let [r, c] = win.window_size();
    let max = (5.0 / 2.0 + 1.0 + 2.0 * 2.0) as usize; // 7.5 -> 7
    assert!(r <= max && c <= max && r > 0 && c > 0, "{r} x {c}");
    for x in points(30, bounds[0], bounds[1]) {
        assert_eq!(win.sample(x).unwrap(), full.sample(x).unwrap());
        assert_eq!(win.gradient(x).unwrap(), full.gradient(x).unwrap());
    }
    // a point beyond the window is an error, not a wrong value
    assert!(matches!(
        win.sample([100.0, 80.0]),
        Err(MapError::Outside { .. })
    ));
}

#[test]
fn unknown_map_lists_available() {
    let specs = [fixture("linear"), fixture("no_value")];
    let err = Maps::read(&specs, &["nope"], FULL).unwrap_err();
    assert!(matches!(err, MapError::Unknown { .. }));
    let msg = err.to_string();
    assert!(
        msg.contains("nope") && msg.contains("linear") && msg.contains("no_value"),
        "{msg}"
    );

    let maps = Maps::read(&specs, &["linear"], FULL).unwrap();
    assert_eq!(maps.get("linear").unwrap().id(), "linear");
    let err = maps.get("no_value").unwrap_err();
    assert!(err.to_string().contains("linear"));
}

#[test]
fn test_map_round_trip() {
    let dir = tempdir();
    let (rows, cols) = (5, 7);
    let values: Vec<f64> = (0..rows * cols).map(|i| i as f64 * 0.5 - 3.0).collect();
    let spec = write_test_map(&dir.join("m.zarr"), &values, rows, cols, 1.5, [-2.0, 4.0]);
    let map = Map::read(&spec, FULL).unwrap();
    for r in 0..rows {
        for c in 0..cols {
            let x = [-2.0 + c as f64 * 1.5, 4.0 + r as f64 * 1.5];
            assert_eq!(map.sample(x).unwrap(), values[r * cols + c], "({r}, {c})");
        }
    }
    std::fs::remove_dir_all(dir).ok();
}

fn tempdir() -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "maps-test-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn float32_no_value_matches_in_f32_precision() {
    use std::sync::Arc;
    use zarrs::array::{ArrayBuilder, data_type};
    let dir = tempdir().join("f32");
    let store = Arc::new(zarrs::filesystem::FilesystemStore::new(&dir).unwrap());
    let mut values = vec![1.0f32; 16];
    values[5] = -9999.9; // row 1, col 1
    let array = ArrayBuilder::new(vec![4, 4], vec![4, 4], data_type::float32(), 0.0f32)
        .build(store, "/")
        .unwrap();
    array.store_metadata().unwrap();
    array
        .store_array_subset(&array.subset_all(), values)
        .unwrap();
    let spec = MapSpec {
        id: "f32map".into(),
        path: dir.clone(),
        units: "1".into(),
        um_per_px: 1.0,
        origin_um: [0.0, 0.0],
        no_value: Some(-9999.9),
    };
    let map = Map::read(&spec, FULL).unwrap();
    assert!(matches!(
        map.sample([1.2, 1.3]),
        Err(MapError::NoValue { .. })
    ));
    assert_eq!(map.sample([2.5, 2.5]).unwrap(), 1.0);
    std::fs::remove_dir_all(dir.parent().unwrap()).ok();
}

/// A map of `f(x) = x + 2 y` made by `Map::from_fn` holds the values of `f` at the pixel centres
/// of the window `Map::read` would read: bilinear samples and gradients are those of `f` inside
/// the window, and it is the window, values included, of the same raster read from Zarr.
#[test]
fn from_fn_samples_exactly_at_pixel_centres() {
    let f = |x: [f64; 2]| x[0] + 2.0 * x[1];
    let (um_per_px, origin_um, shape) = (0.5, [-0.5, 0.25], [40, 30]);
    let bounds = [[3.0, 4.0], [9.0, 7.0]];
    let map = Map::from_fn("linear", um_per_px, origin_um, shape, bounds, f);
    assert_eq!(map.id(), "linear");
    for x in points(200, bounds[0], bounds[1]) {
        let v = map.sample(x).unwrap();
        assert!((v - f(x)).abs() <= 1e-12, "{x:?}: {v} vs {}", f(x));
        let g = map.gradient(x).unwrap();
        assert!(
            (g[0] - 1.0).abs() <= 1e-12 && (g[1] - 2.0).abs() <= 1e-12,
            "{x:?}: {g:?}"
        );
    }
    // The same pixels written to Zarr and read over the same bounds.
    let mut values = Vec::with_capacity(shape[0] * shape[1]);
    for r in 0..shape[0] {
        for c in 0..shape[1] {
            let x = [
                origin_um[0] + c as f64 * um_per_px,
                origin_um[1] + r as f64 * um_per_px,
            ];
            values.push(f(x));
        }
    }
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("from_fn_samples_exactly");
    if dir.exists() {
        std::fs::remove_dir_all(&dir).unwrap();
    }
    let spec = write_test_map(&dir, &values, shape[0], shape[1], um_per_px, origin_um);
    let read = Map::read(&spec, bounds).unwrap();
    assert_eq!(map.window_size(), read.window_size());
    for x in points(200, bounds[0], bounds[1]) {
        assert_eq!(map.sample(x).unwrap(), read.sample(x).unwrap(), "{x:?}");
        assert_eq!(map.gradient(x).unwrap(), read.gradient(x).unwrap(), "{x:?}");
    }
    // Beyond the window (inside the raster) there is no value.
    assert!(matches!(
        map.sample([12.0, 15.0]),
        Err(MapError::Outside { .. })
    ));
}

/// `Maps::insert` adds a map under its id, or replaces the map of that id.
#[test]
fn insert_adds_or_replaces_by_id() {
    let mut maps = Maps::default();
    let window = [[0.0, 0.0], [4.0, 4.0]];
    maps.insert(Map::from_fn("a", 1.0, [0.0, 0.0], [8, 8], window, |_| 1.0));
    maps.insert(Map::from_fn("b", 1.0, [0.0, 0.0], [8, 8], window, |_| 2.0));
    maps.insert(Map::from_fn("a", 1.0, [0.0, 0.0], [8, 8], window, |_| 3.0));
    assert_eq!(maps.get("a").unwrap().sample([1.0, 1.0]).unwrap(), 3.0);
    assert_eq!(maps.get("b").unwrap().sample([1.0, 1.0]).unwrap(), 2.0);
}
