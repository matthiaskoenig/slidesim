//! The inputs of a simulation ([`Inputs`]): the study, its scene, the cell model and, optionally,
//! a mesh in place of the meshing of the scene.

use std::path::Path;

use anyhow::{Context, Result};
use contracts::{Scene, Study, check_cells, check_maps};
use mesher::TaggedMesh;

use crate::coupled::linked_fields;

/// What a [`crate::Simulation`] is made of.
#[derive(Debug, Clone)]
pub struct Inputs {
    /// The `study/1`.
    pub study: Study,
    /// The `scene/1` of the study.
    pub scene: Scene,
    /// The SBML of the cell model of `study.cells` (none without `cells`).
    pub model_sbml: Option<String>,
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
    ///
    /// Errors: those of [`Study::load`], [`Scene::load`], [`check_maps`] and [`check_cells`], the
    /// fields linked to the cells, and a cell model which cannot be read.
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
