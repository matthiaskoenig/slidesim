//! Raster maps read from Zarr v3 arrays: windowed reads, bilinear samples and gradients.
//!
//! A map is a 2D float array (rows = y, cols = x). Pixel (row `r`, col `c`) is centred at
//! `origin_um + [c, r] * um_per_px`. Each process reads only the window of the raster that
//! covers its own cells, so this crate depends neither on PETSc nor on MPI.

use std::collections::BTreeMap;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use contracts::MapSpec;
use zarrs::array::{Array, ArrayBuilder, ArraySubset, data_type};
use zarrs::filesystem::FilesystemStore;

/// Errors of reading and sampling maps; each names the map and, where there is one, the point.
#[derive(Debug, thiserror::Error)]
pub enum MapError {
    #[error("map '{map}': point ({}, {}) um is outside the raster", x[0], x[1])]
    Outside { map: String, x: [f64; 2] },
    #[error(
        "map '{map}': point ({}, {}) um is on a no-value or NaN pixel",
        x[0], x[1]
    )]
    NoValue { map: String, x: [f64; 2] },
    #[error("unknown map '{map}', available: [{}]", available.join(", "))]
    Unknown { map: String, available: Vec<String> },
    #[error("map '{map}': {message}")]
    Format { map: String, message: String },
    #[error("map '{map}': zarr: {message}")]
    Zarr { map: String, message: String },
}

/// Pixels needed around a point for a bilinear sample with gradients: one before the cell for
/// the central difference, the cell's two pixels, one after.
const MARGIN: i64 = 1;

/// One raster window held in memory.
#[derive(Debug, Clone)]
pub struct Map {
    id: String,
    um_per_px: f64,
    origin_um: [f64; 2],
    no_value: Option<f64>,
    /// Size of the whole array (rows, cols).
    shape: [usize; 2],
    /// Window offset into the array (row, col) and size.
    offset: [usize; 2],
    size: [usize; 2],
    /// Window values, row-major; no-value pixels are stored as NaN.
    data: Vec<f64>,
}

fn zarr_err(map: &str, e: impl std::fmt::Display) -> MapError {
    MapError::Zarr {
        map: map.to_string(),
        message: e.to_string(),
    }
}

fn format_err(map: &str, message: impl Into<String>) -> MapError {
    MapError::Format {
        map: map.to_string(),
        message: message.into(),
    }
}

impl Map {
    /// Read the pixels covering `bounds` (`[[xmin, ymin], [xmax, ymax]]` in um) plus the margin
    /// needed for samples and gradients, clipped to the array; only the chunks needed.
    pub fn read(spec: &MapSpec, bounds: [[f64; 2]; 2]) -> Result<Map, MapError> {
        let id = spec.id.as_str();
        if spec.um_per_px.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater)
            || !spec.um_per_px.is_finite()
        {
            return Err(format_err(id, "um_per_px must be positive and finite"));
        }
        let store = Arc::new(FilesystemStore::new(&spec.path).map_err(|e| zarr_err(id, e))?);
        let array = Array::open(store, "/").map_err(|e| zarr_err(id, e))?;
        let shape = array.shape().to_vec();
        if shape.len() != 2 {
            return Err(format_err(
                id,
                format!("expected a 2D array, found {} dimensions", shape.len()),
            ));
        }
        let (rows, cols) = (shape[0] as usize, shape[1] as usize);
        if rows < 2 || cols < 2 {
            return Err(format_err(
                id,
                format!("array {rows} x {cols} is too small"),
            ));
        }
        let is_f32 = *array.data_type() == data_type::float32();
        let is_f64 = *array.data_type() == data_type::float64();
        if !is_f32 && !is_f64 {
            return Err(format_err(id, "data type must be float32 or float64"));
        }

        let window = |axis: usize, n: usize| -> Range<usize> {
            let lo = ((bounds[0][axis] - spec.origin_um[axis]) / spec.um_per_px).floor() as i64;
            let hi = ((bounds[1][axis] - spec.origin_um[axis]) / spec.um_per_px).floor() as i64;
            let a = (lo - MARGIN).clamp(0, n as i64);
            let b = (hi + 1 + MARGIN + 1).clamp(a, n as i64);
            a as usize..b as usize
        };
        let (rr, cr) = (window(1, rows), window(0, cols));
        let size = [rr.len(), cr.len()];
        let mut data = Vec::new();
        if size[0] > 0 && size[1] > 0 {
            let subset = ArraySubset::new_with_ranges(&[
                rr.start as u64..rr.end as u64,
                cr.start as u64..cr.end as u64,
            ]);
            data = if is_f64 {
                array
                    .retrieve_array_subset::<Vec<f64>>(&subset)
                    .map_err(|e| zarr_err(id, e))?
            } else {
                array
                    .retrieve_array_subset::<Vec<f32>>(&subset)
                    .map_err(|e| zarr_err(id, e))?
                    .into_iter()
                    .map(f64::from)
                    .collect()
            };
            if let Some(nv) = spec.no_value {
                for v in &mut data {
                    if *v == nv {
                        *v = f64::NAN;
                    }
                }
            }
        }
        Ok(Map {
            id: spec.id.clone(),
            um_per_px: spec.um_per_px,
            origin_um: spec.origin_um,
            no_value: spec.no_value,
            shape: [rows, cols],
            offset: [rr.start, cr.start],
            size,
            data,
        })
    }

    /// The map id.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The no-value marker of the map, if any.
    pub fn no_value(&self) -> Option<f64> {
        self.no_value
    }

    /// Number of pixels held in the window (rows, cols).
    pub fn window_size(&self) -> [usize; 2] {
        self.size
    }

    fn outside(&self, x: [f64; 2]) -> MapError {
        MapError::Outside {
            map: self.id.clone(),
            x,
        }
    }

    fn no_value_err(&self, x: [f64; 2]) -> MapError {
        MapError::NoValue {
            map: self.id.clone(),
            x,
        }
    }

    /// Pixel (row, col) of the array if it is held in the window.
    fn pixel(&self, r: usize, c: usize) -> Option<f64> {
        let (wr, wc) = (
            r.checked_sub(self.offset[0])?,
            c.checked_sub(self.offset[1])?,
        );
        if wr < self.size[0] && wc < self.size[1] {
            Some(self.data[wr * self.size[1] + wc])
        } else {
            None
        }
    }

    /// Cell index and fractions of `x`: `(row, col, ty, tx)` with the cell spanning pixels
    /// `row..=row+1`, `col..=col+1`.
    fn locate(&self, x: [f64; 2]) -> Result<(usize, usize, f64, f64), MapError> {
        let px = (x[0] - self.origin_um[0]) / self.um_per_px;
        let py = (x[1] - self.origin_um[1]) / self.um_per_px;
        let (maxc, maxr) = ((self.shape[1] - 1) as f64, (self.shape[0] - 1) as f64);
        if !(px >= 0.0 && px <= maxc && py >= 0.0 && py <= maxr) {
            return Err(self.outside(x));
        }
        let c = (px.floor() as usize).min(self.shape[1] - 2);
        let r = (py.floor() as usize).min(self.shape[0] - 2);
        Ok((r, c, py - r as f64, px - c as f64))
    }

    fn value(&self, r: usize, c: usize, x: [f64; 2]) -> Result<f64, MapError> {
        match self.pixel(r, c) {
            None => Err(self.outside(x)),
            Some(v) if v.is_nan() => Err(self.no_value_err(x)),
            Some(v) => Ok(v),
        }
    }

    /// Bilinear interpolation between pixel centres at `x` (um).
    pub fn sample(&self, x: [f64; 2]) -> Result<f64, MapError> {
        let (r, c, ty, tx) = self.locate(x)?;
        let v00 = self.value(r, c, x)?;
        let v01 = self.value(r, c + 1, x)?;
        let v10 = self.value(r + 1, c, x)?;
        let v11 = self.value(r + 1, c + 1, x)?;
        Ok((1.0 - ty) * ((1.0 - tx) * v00 + tx * v01) + ty * ((1.0 - tx) * v10 + tx * v11))
    }

    /// Gradient (per um) at pixel (r, c): central differences, one-sided at the array border.
    fn pixel_gradient(&self, r: usize, c: usize, x: [f64; 2]) -> Result<[f64; 2], MapError> {
        let h = self.um_per_px;
        let diff = |lo: (usize, usize), hi: (usize, usize), n: f64| -> Result<f64, MapError> {
            Ok((self.value(hi.0, hi.1, x)? - self.value(lo.0, lo.1, x)?) / (n * h))
        };
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let (c0, c1) = (c.saturating_sub(1), (c + 1).min(cols - 1));
        let (r0, r1) = (r.saturating_sub(1), (r + 1).min(rows - 1));
        let gx = diff((r, c0), (r, c1), (c1 - c0) as f64)?;
        let gy = diff((r0, c), (r1, c), (r1 - r0) as f64)?;
        Ok([gx, gy])
    }

    /// Gradient `[d/dx, d/dy]` per um at `x`: central differences at the pixel centres
    /// (one-sided at the array border), interpolated bilinearly.
    pub fn gradient(&self, x: [f64; 2]) -> Result<[f64; 2], MapError> {
        let (r, c, ty, tx) = self.locate(x)?;
        let g00 = self.pixel_gradient(r, c, x)?;
        let g01 = self.pixel_gradient(r, c + 1, x)?;
        let g10 = self.pixel_gradient(r + 1, c, x)?;
        let g11 = self.pixel_gradient(r + 1, c + 1, x)?;
        let lerp = |k: usize| {
            (1.0 - ty) * ((1.0 - tx) * g00[k] + tx * g01[k])
                + ty * ((1.0 - tx) * g10[k] + tx * g11[k])
        };
        Ok([lerp(0), lerp(1)])
    }
}

/// The maps a study uses, by id.
#[derive(Debug, Clone, Default)]
pub struct Maps {
    maps: BTreeMap<String, Map>,
}

impl Maps {
    /// Read only the maps named in `ids`, each over `bounds` (see [`Map::read`]).
    pub fn read(specs: &[MapSpec], ids: &[&str], bounds: [[f64; 2]; 2]) -> Result<Maps, MapError> {
        let mut maps = BTreeMap::new();
        for id in ids {
            if maps.contains_key(*id) {
                continue;
            }
            let spec = specs
                .iter()
                .find(|s| s.id == *id)
                .ok_or_else(|| MapError::Unknown {
                    map: (*id).to_string(),
                    available: specs.iter().map(|s| s.id.clone()).collect(),
                })?;
            maps.insert((*id).to_string(), Map::read(spec, bounds)?);
        }
        Ok(Maps { maps })
    }

    /// The map `id`.
    pub fn get(&self, id: &str) -> Result<&Map, MapError> {
        self.maps.get(id).ok_or_else(|| MapError::Unknown {
            map: id.to_string(),
            available: self.maps.keys().cloned().collect(),
        })
    }
}

/// Write a float64 zstd Zarr v3 map of `rows` x `cols` (row-major `values`) at `path` and
/// return its spec (id `test`, units `1`). For the tests of other crates.
#[doc(hidden)]
pub fn write_test_map(
    path: &Path,
    values: &[f64],
    rows: usize,
    cols: usize,
    um_per_px: f64,
    origin_um: [f64; 2],
) -> MapSpec {
    use zarrs::array::codec::ZstdCodec;
    assert_eq!(values.len(), rows * cols, "values must be rows x cols");
    let store = Arc::new(FilesystemStore::new(path).expect("create map store"));
    let chunk = [rows.min(64) as u64, cols.min(64) as u64];
    let array = ArrayBuilder::new(
        vec![rows as u64, cols as u64],
        chunk.to_vec(),
        data_type::float64(),
        0.0f64,
    )
    .bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(3, false))])
    .build(store, "/")
    .expect("build map array");
    array.store_metadata().expect("store map metadata");
    array
        .store_array_subset(&array.subset_all(), values.to_vec())
        .expect("store map values");
    MapSpec {
        id: "test".to_string(),
        path: path.to_path_buf(),
        units: "1".to_string(),
        um_per_px,
        origin_um,
        no_value: None,
    }
}
