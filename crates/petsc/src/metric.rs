//! Metric based adaptation of a DMPlex with Mmg (`DMAdaptMetric`): a new mesh whose edges have
//! the lengths an isotropic metric asks for.

use std::ptr::null_mut;

use crate::{Dm, Error, Result, Vector, c_string, chk, sys};

/// The parameters of Mmg for [`Dm::adapt_metric`], recorded in the metric context of the mesh by
/// [`Dm::isotropic_metric`]. The defaults are PETSc's, but for `no_surf`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MetricOptions {
    /// Keep the boundary as it is (`-dm_plex_metric_no_surf`): Mmg neither splits, collapses
    /// nor moves the edges on the boundary and between cells of different values of the cell
    /// label, so tagged walls and interfaces keep their vertices. On by default.
    pub no_surf: bool,
    /// The largest ratio of the lengths of neighbouring edges (`-dm_plex_metric_gradation_factor`,
    /// Mmg's `hgrad`); PETSc's default 1.3.
    pub gradation: f64,
    /// The largest distance of the boundary of the new mesh from the old one
    /// (`-dm_plex_metric_hausdorff_number`, Mmg's `hausd`), in the unit of the coordinates;
    /// PETSc's default 0.01.
    pub hausdorff: f64,
    /// The verbosity of Mmg from -1 (silent) to 10 (`-dm_plex_metric_verbosity`); PETSc's default
    /// -1.
    pub verbosity: i32,
}

impl Default for MetricOptions {
    fn default() -> MetricOptions {
        MetricOptions {
            no_surf: true,
            gradation: 1.3,
            hausdorff: 0.01,
            verbosity: -1,
        }
    }
}

/// `local` on every process of `dm` if no process has an error, else an error on every process:
/// this process's, or one naming another process. Collective.
fn collective(dm: &Dm, local: Result<()>) -> Result<()> {
    if dm.comm().max_int(local.is_err().into())? > 0 {
        return Err(local.err().unwrap_or_else(|| {
            Error::new(
                sys::PETSC_ERR_ARG_WRONG,
                "an invalid metric on another process",
            )
        }));
    }
    Ok(())
}

impl Dm {
    /// The isotropic metric `1 / h^2` of the edge lengths `sizes` (one per local vertex, in the
    /// order of the depth-0 stratum), for [`Dm::adapt_metric`]: a local vector of P1 on a clone of
    /// this mesh, so that this mesh gets no field. Records `options` (and the isotropy of the
    /// metric) in the metric context of the mesh, which its clones share and
    /// [`Dm::adapt_metric`] reads; the other parameters of Mmg are set to PETSc's defaults
    /// (insertion, swapping and movement of vertices on), whatever the options database says.
    /// Collective.
    ///
    /// Errors (on every process) unless there is one size per vertex, each finite and positive.
    pub fn isotropic_metric(&self, sizes: &[f64], options: &MetricOptions) -> Result<Vector> {
        let vertices = self.depth_stratum(0)?;
        let local = if sizes.len() != vertices.len() {
            Err(Error::new(
                sys::PETSC_ERR_ARG_SIZ,
                format!(
                    "isotropic metric: {} sizes for {} vertices",
                    sizes.len(),
                    vertices.len()
                ),
            ))
        } else if let Some((i, h)) = sizes
            .iter()
            .enumerate()
            .find(|&(_, &h)| !(h.is_finite() && h > 0.0))
        {
            Err(Error::new(
                sys::PETSC_ERR_ARG_OUTOFRANGE,
                format!("isotropic metric: the size {h} of vertex {i} is not finite and positive"),
            ))
        } else {
            Ok(())
        };
        collective(self, local)?;

        let raw = self.as_raw();
        chk!(unsafe { sys::DMPlexMetricSetIsotropic(raw, true) });
        chk!(unsafe { sys::DMPlexMetricSetUniform(raw, false) });
        chk!(unsafe { sys::DMPlexMetricSetNoInsertion(raw, false) });
        chk!(unsafe { sys::DMPlexMetricSetNoSwapping(raw, false) });
        chk!(unsafe { sys::DMPlexMetricSetNoMovement(raw, false) });
        chk!(unsafe { sys::DMPlexMetricSetNoSurf(raw, options.no_surf) });
        chk!(unsafe { sys::DMPlexMetricSetGradationFactor(raw, options.gradation) });
        chk!(unsafe { sys::DMPlexMetricSetHausdorffNumber(raw, options.hausdorff) });
        chk!(unsafe { sys::DMPlexMetricSetVerbosity(raw, options.verbosity as sys::PetscInt) });

        // DMPlexMetricCreate sets the P1 field on the DM it is given: a clone (which shares the
        // topology, the coordinates and the metric context). The vector keeps the clone.
        let clone = self.clone_mesh()?;
        let mut vec = null_mut();
        // SAFETY: `vec` receives a new local vector, whose reference `Vector` takes.
        chk!(unsafe { sys::DMPlexMetricCreate(clone.as_raw(), 0, &mut vec) });
        let mut metric = unsafe { Vector::from_raw(vec) };
        // PETSc's Mmg adaptor reads the value of the i-th vertex at index i of the vector.
        for (i, v) in vertices.clone().enumerate() {
            let dofs = clone.local_dofs(v)?;
            if dofs != (i..i + 1) {
                return Err(Error::new(
                    sys::PETSC_ERR_PLIB,
                    format!("isotropic metric: vertex {v} has the dofs {dofs:?}, not [{i}]"),
                ));
            }
        }
        metric.with_array_mut(|m| {
            for (m, &h) in m.iter_mut().zip(sizes) {
                *m = 1.0 / (h * h);
            }
        })?;
        Ok(metric)
    }

    /// This mesh adapted by Mmg to the metric `metric` of [`Dm::isotropic_metric`]
    /// (`DMAdaptMetric`, with the parameters recorded there): a new DMPlex on the same
    /// communicator. The face label `faces` and the cell label `cells` are carried over to it
    /// with their names and values: an edge of the new mesh on a tagged edge of this one carries
    /// its value, a new cell the value of the cell it lies in. Without `faces`, the boundary is
    /// marked by PETSc's own label; without `cells`, the cells have one value. Faces without a
    /// value of `faces` are not given to Mmg. Collective.
    ///
    /// Mmg is sequential: errors (on every process) if the communicator has more than one
    /// process, or if this mesh has no label `faces` or `cells`.
    pub fn adapt_metric(
        &self,
        metric: &Vector,
        faces: Option<&str>,
        cells: Option<&str>,
    ) -> Result<Dm> {
        let size = self.comm().size();
        if size != 1 {
            return Err(Error::new(
                sys::PETSC_ERR_SUP,
                format!("Mmg adapts a mesh on one process, not on {size}"),
            ));
        }
        let label = |name: Option<&str>| -> Result<sys::DMLabel> {
            let Some(name) = name else {
                return Ok(null_mut());
            };
            let c_name = c_string(name, "label name")?;
            let mut label = null_mut();
            // SAFETY: the label (or null) is borrowed from the DM.
            chk!(unsafe { sys::DMGetLabel(self.as_raw(), c_name.as_ptr(), &mut label) });
            if label.is_null() {
                return Err(Error::new(
                    sys::PETSC_ERR_ARG_WRONG,
                    format!("metric adaptation: the mesh has no label {name:?}"),
                ));
            }
            Ok(label)
        };
        let (faces, cells) = (label(faces)?, label(cells)?);
        let mut raw = null_mut();
        // SAFETY: valid DM, vector and labels (or null); `raw` receives a new DM, or stays null.
        chk!(unsafe { sys::DMAdaptMetric(self.as_raw(), metric.as_raw(), faces, cells, &mut raw) });
        if raw.is_null() {
            return Err(Error::new(
                sys::PETSC_ERR_LIB,
                "DMAdaptMetric returned no mesh",
            ));
        }
        // SAFETY: a new DM, whose reference becomes ours.
        Ok(unsafe { Dm::from_raw(raw) })
    }
}
