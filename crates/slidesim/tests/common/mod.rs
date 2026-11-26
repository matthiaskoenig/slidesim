//! Helpers of the end to end tests: reading the `.vtu` files of a run (raw appended data, 64 bit
//! headers, one piece per process).

#![allow(dead_code)]

use std::path::Path;

/// The raw bytes of every piece of the data array `name` of the `.vtu` file `path`, piece by
/// piece.
fn raw_pieces(path: &Path, name: &str) -> Vec<Vec<u8>> {
    let bytes = std::fs::read(path).unwrap();
    let marker = find(&bytes, b"<AppendedData", 0).expect("appended data");
    let text = String::from_utf8_lossy(&bytes[..marker]).into_owned();
    // The data starts after the underscore which follows the tag.
    let data = marker + find(&bytes, b"_", marker).unwrap() - marker + 1;
    let pattern = format!("Name=\"{name}\"");
    let pieces: Vec<Vec<u8>> = text
        .match_indices(&pattern)
        .map(|(at, _)| {
            let rest = &text[at..];
            let offset: usize = rest[rest.find("offset=\"").unwrap() + 8..]
                .split('"')
                .next()
                .unwrap()
                .parse()
                .unwrap();
            let start = data + offset;
            let len = u64::from_le_bytes(bytes[start..start + 8].try_into().unwrap()) as usize;
            bytes[start + 8..start + 8 + len].to_vec()
        })
        .collect();
    assert!(!pieces.is_empty(), "no array {name} in {}", path.display());
    pieces
}

fn find(bytes: &[u8], what: &[u8], from: usize) -> Option<usize> {
    bytes[from..]
        .windows(what.len())
        .position(|w| w == what)
        .map(|i| i + from)
}

/// The values of the `Float64` data array `name` of every piece of the `.vtu` file `path`.
pub fn vtu_array(path: &Path, name: &str) -> Vec<f64> {
    raw_pieces(path, name)
        .iter()
        .flat_map(|piece| {
            piece
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| f64::from_le_bytes(*b))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// The values of the `Int32` data array `name` of each piece of the `.vtu` file `path`.
fn vtu_ints(path: &Path, name: &str) -> Vec<Vec<i32>> {
    raw_pieces(path, name)
        .iter()
        .map(|piece| {
            piece
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| i32::from_le_bytes(*b))
                .collect()
        })
        .collect()
}

/// The centroid of every cell of the `.vtu` file `path` (of its first two coordinates), in the
/// order of the cell data.
pub fn cell_centroids(path: &Path) -> Vec<[f64; 2]> {
    let positions = raw_pieces(path, "Position");
    let connectivity = vtu_ints(path, "connectivity");
    let offsets = vtu_ints(path, "offsets");
    let mut centroids = Vec::new();
    for ((points, connectivity), offsets) in positions.iter().zip(&connectivity).zip(&offsets) {
        let points: Vec<f64> = points
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| f64::from_le_bytes(*b))
            .collect();
        let mut start = 0;
        for &end in offsets {
            let end = end as usize;
            let vertices = &connectivity[start..end];
            let mut c = [0.0; 2];
            for &v in vertices {
                c[0] += points[3 * v as usize];
                c[1] += points[3 * v as usize + 1];
            }
            centroids.push(c.map(|x| x / vertices.len() as f64));
            start = end;
        }
    }
    centroids
}

/// The cell data `names` of the `.vtu` file `path` by cell, keyed by the centroid of the cell
/// (rounded to 1e-6 um), so that the files of runs on different numbers of processes compare.
pub fn cells_by_centroid(path: &Path, names: &[&str]) -> Vec<([i64; 2], Vec<f64>)> {
    let centroids = cell_centroids(path);
    let columns: Vec<Vec<f64>> = names.iter().map(|n| vtu_array(path, n)).collect();
    let mut cells: Vec<([i64; 2], Vec<f64>)> = centroids
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let key = c.map(|x| (x * 1e6).round() as i64);
            (key, columns.iter().map(|column| column[i]).collect())
        })
        .collect();
    cells.sort_by_key(|(key, _)| *key);
    cells
}

/// The point data `name` of the `.vtu` file `path` by point, keyed by its coordinates (rounded
/// to 1e-6 um), each point once (the pieces share points).
pub fn points_by_position(path: &Path, name: &str) -> Vec<([i64; 2], f64)> {
    let positions = vtu_array(path, "Position");
    let values = vtu_array(path, name);
    assert_eq!(positions.len(), 3 * values.len());
    let mut points: Vec<([i64; 2], f64)> = positions
        .as_chunks::<3>()
        .0
        .iter()
        .zip(&values)
        .map(|(x, &v)| ([x[0], x[1]].map(|x| (x * 1e6).round() as i64), v))
        .collect();
    points.sort_by_key(|a| a.0);
    points.dedup_by_key(|(key, _)| *key);
    points
}

/// The largest difference of `a` and `b` relative to the largest absolute value of both.
pub fn relative_difference(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let scale = a.iter().chain(b).fold(0.0_f64, |m, x| m.max(x.abs()));
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
        / scale
}
