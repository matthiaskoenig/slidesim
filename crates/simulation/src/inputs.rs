//! The inputs of a simulation ([`Inputs`]): the study, its scene, the cell model and, optionally,
//! a mesh in place of the meshing of the scene.

#[cfg(feature = "native")]
use std::path::Path;

#[cfg(feature = "native")]
use anyhow::Context;
use anyhow::{Result, ensure};
#[cfg(not(feature = "native"))]
use cells::KernelLayout;
use contracts::{Scene, Study, check_cells, check_maps};
use mesher::TaggedMesh;

use crate::OnEveryProcess;
use crate::coupled::linked_fields;

/// What a [`crate::Simulation`] is made of.
#[derive(Debug, Clone)]
pub struct Inputs {
    /// The `study/1`.
    pub study: Study,
    /// The `scene/1` of the study.
    pub scene: Scene,
    /// The SBML of the cell model of `study.cells` (none without `cells`), which the run
    /// compiles to native code. With the feature `native`.
    #[cfg(feature = "native")]
    pub model_sbml: Option<String>,
    /// The batched kernel of the cell model of `study.cells` (none without `cells`): the layout
    /// of `kernel.json`, whose `kernel.wasm` the host compiled from the SBML of the study and
    /// links (see `cells::KernelCells`). Without the feature `native` (the browser build).
    #[cfg(not(feature = "native"))]
    pub kernel: Option<KernelLayout>,
    /// The coarse mesh, in place of the mesh of the scene (as `slidesim mesh` writes it as
    /// `mesh.json`): the tags of the scene's mesh, refined `mesh.refinements` times. The scene
    /// gives the rest: its maps, and the zonation map of a `lobules` source, which is computed
    /// from the lobules as without a mesh. Only process 0's mesh is used.
    pub mesh: Option<TaggedMesh>,
}

impl Inputs {
    /// The study of the file `study_path`, its scene and its cell model (the path of
    /// `cells.model` relative to the directory of the study), without a mesh: the scene is
    /// meshed. The study is checked against the scene before the model is read, as
    /// [`crate::Simulation::new`] checks it, so that an error of the study is reported first.
    /// With the feature `native`.
    ///
    /// Errors: those of [`Study::load`], [`Scene::load`], [`check_maps`] and [`check_cells`], the
    /// fields linked to the cells, and a cell model which cannot be read.
    #[cfg(feature = "native")]
    pub fn load(study_path: &Path) -> Result<Inputs> {
        let study = Study::load(study_path).context("cannot load the study")?;
        let scene = Scene::load(&study.scene).context("cannot load the scene")?;
        check(&study, &scene)?;
        let model_sbml = study
            .cells
            .as_ref()
            .map(|spec| {
                let path = study_path
                    .parent()
                    .unwrap_or(Path::new("."))
                    .join(&spec.model);
                std::fs::read_to_string(&path)
                    .with_context(|| format!("cannot read the cell model {}", path.display()))
            })
            .transpose()?;
        Ok(Inputs {
            study,
            scene,
            model_sbml,
            mesh: None,
        })
    }

    /// Whether the inputs have the cell model of `study.cells`.
    fn has_cell_model(&self) -> bool {
        #[cfg(feature = "native")]
        return self.model_sbml.is_some();
        #[cfg(not(feature = "native"))]
        return self.kernel.is_some();
    }
}

/// The error of a study with cells whose inputs have no cell model.
#[cfg(feature = "native")]
const NO_CELL_MODEL: &str = "cells: the inputs have no cell model";
#[cfg(not(feature = "native"))]
const NO_CELL_MODEL: &str = "cells: the inputs have no kernel of the cell model";

/// Checks the inputs before anything is made, as [`crate::Simulation::new`] does: a study of at
/// least one field and, with cells, a cell model (both errors on every process, with
/// [`OnEveryProcess`]: every process has the inputs), then [`check`].
pub(crate) fn check_inputs(inputs: &Inputs) -> Result<()> {
    ensure!(
        !inputs.study.fields.is_empty(),
        OnEveryProcess("the study has no fields")
    );
    check(&inputs.study, &inputs.scene)?;
    ensure!(
        inputs.study.cells.is_none() || inputs.has_cell_model(),
        OnEveryProcess(NO_CELL_MODEL)
    );
    Ok(())
}

/// Checks the study against its scene before anything is solved: the maps the study names are
/// in the scene ([`check_maps`]), its `cells` section against the study and the scene
/// ([`check_cells`]) and the fields linked to the cells ([`linked_fields`]).
pub(crate) fn check(study: &Study, scene: &Scene) -> Result<()> {
    check_maps(study, scene)?;
    check_cells(study, scene)?;
    if let Some(spec) = &study.cells {
        linked_fields(study, spec)?;
    }
    Ok(())
}
