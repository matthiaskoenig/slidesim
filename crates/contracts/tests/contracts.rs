use std::path::{Path, PathBuf};

use contracts::{Condition, Scene, Study};

fn data(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/contracts")
        .join(name)
}

fn study_err(name: &str) -> String {
    Study::load(data(name)).unwrap_err().to_string()
}

#[test]
fn study_round_trip() {
    let text = std::fs::read_to_string(data("study-minimal.json")).unwrap();
    let original: serde_json::Value = serde_json::from_str(&text).unwrap();
    let study: Study = serde_json::from_str(&text).unwrap();
    assert_eq!(serde_json::to_value(&study).unwrap(), original);
    assert_eq!(
        study.fields[0].boundaries[0].condition,
        Condition::Dirichlet { value: 0.1 }
    );
    assert_eq!(
        study.fields[0].boundaries[1].condition,
        Condition::Flux { value: 1.0 }
    );
}

#[test]
fn load_resolves_paths_against_file_directory() {
    let study = Study::load(data("study-minimal.json")).unwrap();
    assert!(study.scene.is_absolute());
    assert!(
        study
            .scene
            .ends_with("testdata/contracts/scene-minimal.json")
    );
    let scene = Scene::load(data("scene-minimal.json")).unwrap();
    assert!(scene.mesh.path.is_absolute());
    assert!(scene.mesh.path.ends_with("testdata/contracts/mesh.msh"));
}

#[test]
fn study_version() {
    assert_eq!(
        study_err("study-bad-version.json"),
        "unsupported contract version 'study/2', expected 'study/1'"
    );
}

#[test]
fn scene_version() {
    let e = Scene::load(data("scene-bad-version.json")).unwrap_err();
    assert_eq!(
        e.to_string(),
        "unsupported contract version 'scene/2', expected 'scene/1'"
    );
}

#[test]
fn length_unit() {
    let e = Scene::load(data("scene-bad-unit.json")).unwrap_err();
    assert_eq!(e.to_string(), "unsupported length unit 'mm', expected 'um'");
}

#[test]
fn negative_diffusion() {
    assert_eq!(
        study_err("study-negative-diffusion.json"),
        "field 'oxygen': diffusion must be > 0, got -1"
    );
}

#[test]
fn bad_outputs() {
    assert_eq!(
        study_err("study-bad-outputs.json"),
        "time.outputs must be sorted and within [0, end]"
    );
}

#[test]
fn negative_decay() {
    assert_eq!(
        study_err("study-negative-decay.json"),
        "field 'oxygen': decay must be >= 0, got -1"
    );
}

#[test]
fn zero_dt() {
    assert_eq!(
        study_err("study-zero-dt.json"),
        "time.dt must be > 0, got 0"
    );
}

#[test]
fn zero_end() {
    assert_eq!(
        study_err("study-zero-end.json"),
        "time.end must be > 0, got 0"
    );
}

#[test]
fn empty_fields() {
    assert_eq!(
        study_err("study-empty-fields.json"),
        "fields must not be empty"
    );
}

#[test]
fn duplicate_field_id() {
    assert_eq!(
        study_err("study-duplicate-field.json"),
        "duplicate field id 'oxygen'"
    );
}

#[test]
fn duplicate_boundary_tag() {
    assert_eq!(
        study_err("study-duplicate-tag.json"),
        "field 'oxygen': duplicate boundary tag 'inlet'"
    );
}

#[test]
fn unknown_keys_are_rejected() {
    let e = study_err("study-unknown-key.json");
    assert!(
        e.contains("study-unknown-key.json") && e.contains("bogus"),
        "{e}"
    );
}

#[test]
fn io_error_names_path() {
    let e = Study::load(data("missing.json")).unwrap_err();
    assert!(e.to_string().contains("missing.json"));
}

#[test]
fn schemas_are_current() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../schemas");
    for (name, schema) in [
        ("scene-1.schema.json", contracts::schema_scene()),
        ("study-1.schema.json", contracts::schema_study()),
    ] {
        let text = std::fs::read_to_string(dir.join(name)).unwrap_or_default();
        let on_disk: serde_json::Value = serde_json::from_str(&text).unwrap_or_default();
        assert_eq!(
            on_disk, schema,
            "schemas/{name} is stale, run `cargo run -p contracts --example write-schemas`"
        );
    }
}
