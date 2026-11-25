use std::path::{Path, PathBuf};

use contracts::{Coefficient, Condition, Scene, Study, Table, check_maps};

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
fn bad_field_id() {
    assert_eq!(
        study_err("study-bad-field-id.json"),
        "field id 'oxygen-2' must match ^[A-Za-z][A-Za-z0-9_]*$"
    );
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

#[test]
fn boundary_needs_exactly_one_condition() {
    for f in ["study-both-conditions.json", "study-no-condition.json"] {
        let e = study_err(f);
        assert!(
            e.contains("boundary 'inlet': exactly one of 'dirichlet' and 'flux' is required"),
            "{f}: {e}"
        );
    }
}

#[test]
fn bad_dimension() {
    let e = Scene::load(data("scene-bad-dimension.json")).unwrap_err();
    assert_eq!(e.to_string(), "dimension must be 2 or 3, got 4");
}

fn schema_accepts(schema: serde_json::Value, file: &str) -> bool {
    let text = std::fs::read_to_string(data(file)).unwrap();
    let doc: serde_json::Value = serde_json::from_str(&text).unwrap();
    jsonschema::validator_for(&schema).unwrap().is_valid(&doc)
}

#[test]
fn schemas_match_load() {
    assert!(schema_accepts(
        contracts::schema_study(),
        "study-minimal.json"
    ));
    assert!(schema_accepts(
        contracts::schema_scene(),
        "scene-minimal.json"
    ));
    assert!(schema_accepts(
        contracts::schema_study(),
        "study-perfusion.json"
    ));
    assert!(schema_accepts(contracts::schema_scene(), "scene-maps.json"));
    // Strictly increasing table values and unique tags are checked by `load` only.
    for f in [
        "study-table-negative-diffusion.json",
        "study-perfusion-no-boundary.json",
        "study-bad-version.json",
        "study-both-conditions.json",
        "study-no-condition.json",
        "study-empty-fields.json",
        "study-zero-diffusion.json",
        "study-negative-diffusion.json",
        "study-negative-decay.json",
        "study-zero-dt.json",
        "study-zero-end.json",
        "study-unknown-key.json",
        "study-bad-field-id.json",
    ] {
        assert!(
            !schema_accepts(contracts::schema_study(), f),
            "schema accepts {f}"
        );
        assert!(Study::load(data(f)).is_err(), "load accepts {f}");
    }
    for f in ["scene-bad-version.json", "scene-bad-dimension.json"] {
        assert!(
            !schema_accepts(contracts::schema_scene(), f),
            "schema accepts {f}"
        );
        assert!(Scene::load(data(f)).is_err(), "load accepts {f}");
    }
}

/// The studies of `testdata/studies` (`<name>/study.json`, run end to end by the `slidesim`
/// tests; `m2_real_slide` has a test of its own) are valid: by the
/// schemas, and by `Study::load` and `Scene::load`, which resolve the scene and the mesh.
#[test]
fn example_studies_are_valid() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/studies");
    let validates = |schema: serde_json::Value, path: &Path| {
        let doc: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        jsonschema::validator_for(&schema).unwrap().is_valid(&doc)
    };
    let mut studies: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.file_name().unwrap() != "m2_real_slide")
        .collect();
    studies.sort();
    assert!(!studies.is_empty());
    for study in studies {
        let study_path = study.join("study.json");
        let study = study.display();
        assert!(validates(contracts::schema_study(), &study_path), "{study}");
        let loaded = Study::load(&study_path).unwrap();
        assert!(
            validates(contracts::schema_scene(), &loaded.scene),
            "{study}"
        );
        let scene = Scene::load(&loaded.scene).unwrap();
        assert!(scene.mesh.path.is_file(), "{}", scene.mesh.path.display());
    }
}

/// The studies of the end-to-end run on a real slide (`testdata/studies/m2_real_slide`,
/// `<name>.study.json`) are valid by the schema. Their scene is written by
/// `scripts/e2e/m2_real_slide.sh` from slides outside the repository, so it is not loaded here.
#[test]
fn real_slide_studies_are_valid() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/studies/m2_real_slide");
    let schema = jsonschema::validator_for(&contracts::schema_study()).unwrap();
    let mut names = Vec::new();
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_str().unwrap().to_owned();
        let doc: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert!(schema.is_valid(&doc), "{name}");
        names.push(name);
    }
    names.sort();
    assert_eq!(
        names,
        [
            "baseline.study.json",
            "calibrated.study.json",
            "radial10.study.json",
            "whole_blood.study.json"
        ]
    );
}

#[test]
fn scene_maps_load_and_resolve() {
    let scene = Scene::load(data("scene-maps.json")).unwrap();
    assert_eq!(scene.maps.len(), 1);
    let m = &scene.maps[0];
    assert_eq!(m.id, "zonation");
    assert!(m.path.is_absolute());
    assert!(m.path.ends_with("testdata/contracts/maps/zonation.tif"));
    assert_eq!(m.um_per_px, 1.5);
    assert_eq!(m.no_value, Some(-1.0));
}

#[test]
fn study_perfusion_round_trip() {
    let study = Study::load(data("study-perfusion.json")).unwrap();
    assert_eq!(
        study.fields[0].diffusion,
        Coefficient::Table(Table {
            map: "zonation".into(),
            points: vec![[0.0, 500.0], [1.0, 900.0]]
        })
    );
    assert_eq!(study.fields[0].porosity, Coefficient::Constant(0.15));
    assert!(study.fields[0].advection);
    let p = study.perfusion.as_ref().unwrap();
    assert_eq!(p.viscosity, 0.0024);
    assert_eq!(p.direction.g0, 0.00043);
    assert_eq!(p.boundaries.len(), 2);
    let back: Study = serde_json::from_value(serde_json::to_value(&study).unwrap()).unwrap();
    assert_eq!(back, study);
    let reloaded =
        Study::from_value(serde_json::to_value(&study).unwrap(), &data("x.json")).unwrap();
    assert_eq!(reloaded.perfusion, study.perfusion);
}

#[test]
fn table_eval() {
    let c = Coefficient::Table(Table {
        map: "m".into(),
        points: vec![[0.0, 1.0], [1.0, 3.0]],
    });
    assert_eq!(c.eval(-1.0), 1.0);
    assert_eq!(c.eval(0.5), 2.0);
    assert_eq!(c.eval(2.0), 3.0);
    assert_eq!(c.maps(), Some("m"));
    assert_eq!(Coefficient::Constant(4.0).eval(9.0), 4.0);
    assert_eq!(Coefficient::Constant(4.0).maps(), None);
}

#[test]
fn table_not_increasing() {
    let e = study_err("study-table-not-increasing.json");
    assert!(
        e.contains("diffusion") && e.contains("strictly increasing"),
        "{e}"
    );
}

#[test]
fn table_negative_diffusion() {
    let e = study_err("study-table-negative-diffusion.json");
    assert!(e.contains("diffusion") && e.contains("> 0"), "{e}");
}

#[test]
fn perfusion_without_boundary() {
    let e = study_err("study-perfusion-no-boundary.json");
    assert!(e.contains("perfusion.boundaries"), "{e}");
}

#[test]
fn perfusion_duplicate_tag() {
    let e = study_err("study-perfusion-duplicate-tag.json");
    assert!(e.contains("duplicate") && e.contains("portal_wall"), "{e}");
}

#[test]
fn unknown_map_lists_available() {
    let scene = Scene::load(data("scene-maps.json")).unwrap();
    let mut study = Study::load(data("study-perfusion.json")).unwrap();
    check_maps(&study, &scene).unwrap();
    study.perfusion.as_mut().unwrap().direction.map = "zonation2".into();
    let e = check_maps(&study, &scene).unwrap_err().to_string();
    assert!(e.contains("zonation2") && e.contains("zonation"), "{e}");
}

#[test]
fn milestone_1_studies_still_load() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata");
    // The studies of milestone 1: later studies (lobule_toy) use maps and perfusion.
    let files = [
        root.join("contracts/study-minimal.json"),
        root.join("studies/tissue_oxygen/study.json"),
    ];
    for f in files {
        let s = Study::load(&f).unwrap_or_else(|e| panic!("{}: {e}", f.display()));
        assert!(s.perfusion.is_none());
        for field in &s.fields {
            assert!(matches!(field.diffusion, Coefficient::Constant(_)));
            assert_eq!(field.porosity, Coefficient::Constant(1.0));
            assert!(!field.advection);
        }
    }
}

/// The JSON of the file `name` of `testdata/contracts`.
fn json(name: &str) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(data(name)).unwrap()).unwrap()
}

/// The error of `Study::from_value` for `study-perfusion.json` changed by `change`.
fn perfusion_study_err(change: impl FnOnce(&mut serde_json::Value)) -> String {
    let mut value = json("study-perfusion.json");
    change(&mut value);
    Study::from_value(value, &data("study-perfusion.json"))
        .unwrap_err()
        .to_string()
}

#[test]
fn table_needs_two_points() {
    let e = perfusion_study_err(|v| {
        v["fields"][0]["diffusion"]["points"] = serde_json::json!([[0.0, 500.0]]);
    });
    assert_eq!(
        e,
        "field 'tracer': diffusion: table needs at least 2 points"
    );
}

#[test]
fn decay_table_is_nonnegative() {
    // A table output of 0 is a valid decay, a negative one is not.
    let mut value = json("study-perfusion.json");
    value["fields"][0]["decay"] =
        serde_json::json!({"map": "zonation", "points": [[0.0, 0.0], [1.0, 0.5]]});
    Study::from_value(value, &data("study-perfusion.json")).unwrap();
    let e = perfusion_study_err(|v| {
        v["fields"][0]["decay"] =
            serde_json::json!({"map": "zonation", "points": [[0.0, 0.1], [1.0, -0.5]]});
    });
    assert_eq!(e, "field 'tracer': decay must be >= 0, got -0.5");
}

#[test]
fn advection_needs_perfusion() {
    let e = perfusion_study_err(|v| {
        v.as_object_mut().unwrap().remove("perfusion");
    });
    assert_eq!(e, "field 'tracer': advection needs a 'perfusion' section");
    // Without advection the same study is valid.
    let mut value = json("study-perfusion.json");
    value.as_object_mut().unwrap().remove("perfusion");
    value["fields"][0]["advection"] = serde_json::Value::from(false);
    Study::from_value(value, &data("study-perfusion.json")).unwrap();
}

#[test]
fn scene_map_rules() {
    let err = |change: &dyn Fn(&mut serde_json::Value)| {
        let mut value = json("scene-maps.json");
        change(&mut value);
        Scene::from_value(value, &data("scene-maps.json"))
            .unwrap_err()
            .to_string()
    };
    assert_eq!(
        err(&|v| v["maps"][0]["id"] = "2zonation".into()),
        "map id '2zonation' must match ^[A-Za-z][A-Za-z0-9_]*$"
    );
    assert_eq!(
        err(&|v| v["maps"][0]["id"] = "zonation-1".into()),
        "map id 'zonation-1' must match ^[A-Za-z][A-Za-z0-9_]*$"
    );
    assert_eq!(
        err(&|v| {
            let map = v["maps"][0].clone();
            v["maps"].as_array_mut().unwrap().push(map);
        }),
        "duplicate map id 'zonation'"
    );
    for px in [0.0, -1.5] {
        assert_eq!(
            err(&|v| v["maps"][0]["um_per_px"] = px.into()),
            format!("map 'zonation': um_per_px must be > 0, got {px}")
        );
    }
    // Without no_value the map is valid: NaN pixels are the only ones without data.
    let mut value = json("scene-maps.json");
    value["maps"][0].as_object_mut().unwrap().remove("no_value");
    let scene = Scene::from_value(value, &data("scene-maps.json")).unwrap();
    assert_eq!(scene.maps[0].no_value, None);
}
