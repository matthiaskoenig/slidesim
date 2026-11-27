use std::path::{Path, PathBuf};

use contracts::{
    Coefficient, Condition, MeshSource, Polygons, ResultMeta, Scene, Study, Table, check_maps,
};

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
    let path = scene.mesh.path().unwrap();
    assert!(path.is_absolute());
    assert!(path.ends_with("testdata/contracts/mesh.msh"));
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
        ("polygons-1.schema.json", contracts::schema_polygons()),
        ("study-1.schema.json", contracts::schema_study()),
        ("result-1.schema.json", contracts::schema_result()),
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
    for f in ["scene-lobules.json", "scene-polygons.json"] {
        assert!(schema_accepts(contracts::schema_scene(), f), "{f}");
    }
    assert!(schema_accepts(
        contracts::schema_polygons(),
        "polygons-minimal.geojson"
    ));
    for f in [
        "polygons-bad-kind.geojson",
        "polygons-bad-value.geojson",
        "scene-lobules-bad-jitter.json",
        "scene-lobules-bad-radius.json",
    ] {
        let schema = if f.starts_with("polygons") {
            contracts::schema_polygons()
        } else {
            contracts::schema_scene()
        };
        assert!(!schema_accepts(schema, f), "schema accepts {f}");
    }
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

/// The studies of `testdata/studies` (`<name>/study.json` and `<name>/<variant>.study.json`, run
/// end to end by the `slidesim` tests; `m2_real_slide` and `m3_real_slide` have a test of their
/// own) are valid: by the
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
        .filter(|path| {
            !path
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .ends_with("_real_slide")
        })
        .flat_map(|dir| std::fs::read_dir(dir).unwrap())
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            let name = path.file_name().unwrap().to_str().unwrap();
            name == "study.json" || name.ends_with(".study.json")
        })
        .collect();
    studies.sort();
    assert!(studies.len() >= 5, "{studies:?}");
    for study_path in studies {
        let study = study_path.display();
        assert!(validates(contracts::schema_study(), &study_path), "{study}");
        let loaded = Study::load(&study_path).unwrap();
        assert!(
            validates(contracts::schema_scene(), &loaded.scene),
            "{study}"
        );
        let scene = Scene::load(&loaded.scene).unwrap();
        let path = scene.mesh.path().unwrap();
        assert!(path.is_file(), "{}", path.display());
    }
}

/// The studies of the end-to-end runs on a real slide (`testdata/studies/m2_real_slide`,
/// `<name>.study.json`, and `testdata/studies/m3_real_slide/spt.study.json`) are valid by the
/// schema. Their scenes are written from slides outside the repository (by
/// `scripts/e2e/m2_real_slide.sh`, and as `docs/results/2026-11-m3-performance.md` describes), so
/// they are not loaded here.
#[test]
fn real_slide_studies_are_valid() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/studies/m3_real_slide/spt.study.json");
    let doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let schema = jsonschema::validator_for(&contracts::schema_study()).unwrap();
    assert!(schema.is_valid(&doc), "m3_real_slide/spt.study.json");
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

fn cells_study(cells: serde_json::Value) -> Study {
    let mut v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(data("study-perfusion.json")).unwrap())
            .unwrap();
    v["cells"] = cells;
    Study::from_value(v, &data("study-perfusion.json")).unwrap()
}

fn cells_json() -> serde_json::Value {
    serde_json::json!({
        "model": "cell.xml",
        "solver": "engine",
        "time_unit_s": 60.0,
        "areas": ["a", "b"],
        "species": [{"species": "S", "field": "tracer"}],
        "volumes": {"blood": "Vb", "cells": "Vc"},
        "parameters": [
            {"id": "pattern", "value": 7.0,
             "replace": {"map": "zonation", "above": 0.5, "value": 6.0}},
            {"id": "position", "value": {"map": "zonation", "points": [[0.0, 0.0], [1.0, 1.0]]}}
        ],
        "split_step": 1.0,
        "rtol": 1e-5,
        "atol": 1e-8,
        "outputs": ["flux"]
    })
}

#[test]
fn cells_round_trip() {
    let cells = cells_json();
    let study = cells_study(cells.clone());
    let spec = study.cells.as_ref().unwrap();
    assert_eq!(spec.solver, contracts::CellSolverKind::Engine);
    assert_eq!(serde_json::to_value(spec).unwrap(), cells);
}

#[test]
fn cells_defaults() {
    let study = cells_study(serde_json::json!({
        "model": "cell.xml",
        "time_unit_s": 1.0,
        "species": [],
        "volumes": {"blood": "Vb", "cells": "Vc"},
        "parameters": [],
        "split_step": 0.5
    }));
    let spec = study.cells.unwrap();
    assert_eq!(spec.solver, contracts::CellSolverKind::Batched);
    assert_eq!(spec.areas, None);
    assert_eq!((spec.rtol, spec.atol), (1e-6, 1e-9));
    assert!(spec.outputs.is_empty());
}

#[test]
fn cells_unknown_field_lists_fields() {
    let scene = Scene::load(data("scene-maps.json")).unwrap();
    let mut c = cells_json();
    c["species"][0]["field"] = "nope".into();
    let e = contracts::check_cells(&cells_study(c), &scene)
        .unwrap_err()
        .to_string();
    assert!(e.contains("nope") && e.contains("tracer"), "{e}");
}

#[test]
fn cells_unknown_map_lists_maps() {
    let scene = Scene::load(data("scene-maps.json")).unwrap();
    let mut c = cells_json();
    c["parameters"][0]["replace"]["map"] = "necrosis".into();
    let e = contracts::check_cells(&cells_study(c), &scene)
        .unwrap_err()
        .to_string();
    assert!(e.contains("necrosis") && e.contains("zonation"), "{e}");
    let mut c = cells_json();
    c["parameters"][1]["value"]["map"] = "gone".into();
    let e = contracts::check_cells(&cells_study(c), &scene)
        .unwrap_err()
        .to_string();
    assert!(e.contains("gone") && e.contains("zonation"), "{e}");
}

#[test]
fn cells_split_step_positive() {
    let scene = Scene::load(data("scene-maps.json")).unwrap();
    contracts::check_cells(&cells_study(cells_json()), &scene).unwrap();
    for bad in [0.0, -1.0] {
        let mut c = cells_json();
        c["split_step"] = bad.into();
        let e = contracts::check_cells(&cells_study(c), &scene)
            .unwrap_err()
            .to_string();
        assert!(e.contains("split_step"), "{e}");
    }
    // time.dt must not exceed split_step
    let study = cells_study(cells_json());
    let dt = study.time.dt;
    let mut c = cells_json();
    c["split_step"] = (dt / 2.0).into();
    let e = contracts::check_cells(&cells_study(c), &scene)
        .unwrap_err()
        .to_string();
    assert!(e.contains("time.dt") && e.contains("split_step"), "{e}");
}

#[test]
fn lobules_scene_loads_with_defaults() {
    let dir = std::env::temp_dir().join(format!("slidesim-lobules-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("scene.json");
    std::fs::write(
        &file,
        r#"{"version": "scene/1", "dimension": 2, "units": {"length": "um"},
            "mesh": {"format": "lobules",
                     "centre": {"tag": "c"}, "corner": {"tag": "p"},
                     "outer_tag": "o", "area_tag": "a",
                     "zonation": {"map": "zonation"}}}"#,
    )
    .unwrap();
    let scene = Scene::load(&file).unwrap();
    let MeshSource::Lobules(spec) = &scene.mesh else {
        panic!("not lobules: {:?}", scene.mesh)
    };
    assert_eq!(spec.lobule_radius_um, 300.0);
    assert_eq!(spec.rings, 0);
    assert_eq!(spec.jitter, 0.0);
    assert_eq!(spec.seed, 0);
    assert_eq!(spec.centre.radius_um, 30.0);
    assert_eq!(spec.corner.radius_um, 25.0);
    assert_eq!(spec.size.wall_um, 5.0);
    assert_eq!(spec.size.interior_um, 20.0);
    assert_eq!(spec.zonation.um_per_px, 2.0);
    assert_eq!(scene.map_ids(), ["zonation"]);
    assert!(scene.mesh.path().is_none());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn lobules_scene_full_round_trip() {
    let scene = Scene::load(data("scene-lobules.json")).unwrap();
    let text = std::fs::read_to_string(data("scene-lobules.json")).unwrap();
    let original: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(serde_json::to_value(&scene).unwrap(), original);
}

#[test]
fn lobule_size_field() {
    let scene = Scene::load(data("scene-lobules.json")).unwrap();
    let MeshSource::Lobules(spec) = &scene.mesh else {
        panic!()
    };
    let f = spec.size_field();
    assert_eq!(
        (f.min_um, f.max_um, f.dist_min_um, f.dist_max_um),
        (5.0, 20.0, 10.0, 200.0)
    );
    assert_eq!(f.walls, ["central_wall", "portal_wall"]);
}

#[test]
fn polygons_scene_resolves_its_path() {
    let scene = Scene::load(data("scene-polygons.json")).unwrap();
    let MeshSource::Polygons { path, size } = &scene.mesh else {
        panic!()
    };
    assert!(path.is_absolute());
    assert!(path.ends_with("testdata/contracts/polygons-minimal.geojson"));
    assert!(path.is_file());
    assert_eq!(size.walls, ["central_wall"]);
    assert!(scene.map_ids().is_empty());
}

#[test]
fn gmsh_scenes_still_load() {
    let mut n = 0;
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata");
    let mut stack = vec![root];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(dir).unwrap() {
            let p = entry.unwrap().path();
            let name = p.file_name().unwrap().to_string_lossy().to_string();
            if p.is_dir() {
                stack.push(p);
            } else if name.starts_with("scene") && name.ends_with(".json") && !name.contains("bad")
            {
                let Ok(scene) = Scene::load(&p) else {
                    panic!("{} does not load", p.display())
                };
                if let MeshSource::Gmsh { path } = &scene.mesh {
                    assert!(path.is_absolute());
                    n += 1;
                }
            }
        }
    }
    assert!(n >= 3, "only {n} gmsh scenes found");
}

#[test]
fn lobule_parameters_out_of_range() {
    for (file, key) in [
        ("scene-lobules-bad-radius.json", "lobule_radius_um"),
        ("scene-lobules-bad-jitter.json", "jitter"),
        ("scene-lobules-bad-veins-touch.json", "centre.radius_um"),
        ("scene-lobules-bad-corner.json", "corner.radius_um"),
        ("scene-lobules-bad-size.json", "size.wall_um"),
        ("scene-lobules-bad-map-clash.json", "zonation.map"),
        ("scene-lobules-bad-tags.json", "area_tag"),
    ] {
        let e = Scene::load(data(file)).unwrap_err();
        assert!(matches!(e, contracts::ContractError::Invalid(_)), "{file}");
        assert!(e.to_string().contains(key), "{file}: {e}");
    }
}

#[test]
fn size_field_rules() {
    let base = serde_json::from_str::<serde_json::Value>(
        &std::fs::read_to_string(data("scene-polygons.json")).unwrap(),
    )
    .unwrap();
    let load = |key: &str, v: f64| {
        let mut d = base.clone();
        d["mesh"]["size"][key] = serde_json::json!(v);
        Scene::from_value(d, &data("scene-polygons.json"))
    };
    for (key, v) in [
        ("min_um", 30.0),
        ("min_um", -1.0),
        ("max_um", 1.0),
        ("dist_min_um", 200.0),
        ("dist_min_um", -1.0),
        ("dist_max_um", 10.0),
    ] {
        let e = load(key, v).unwrap_err().to_string();
        assert!(e.contains("size."), "{key}={v}: {e}");
        assert!(e.contains(key), "{key}={v}: {e}");
    }
    assert!(load("dist_min_um", 0.0).is_ok());
}

fn polygons_err(name: &str) -> String {
    Polygons::load(data(name)).unwrap_err().to_string()
}

#[test]
fn polygons_load() {
    let p = Polygons::load(data("polygons-minimal.geojson")).unwrap();
    assert_eq!(p.areas.len(), 2);
    assert_eq!(p.edges.len(), 3);
    assert_eq!((p.areas[0].tag.as_str(), p.areas[0].value), ("tissue", 1));
    assert_eq!(p.areas[0].rings.len(), 1);
    assert_eq!((p.areas[1].tag.as_str(), p.areas[1].value), ("necrotic", 2));
    assert_eq!(p.areas[1].rings.len(), 2);
    assert_eq!(p.edges[0].tag, "central_wall");
    assert_eq!(p.edges[0].points[1], [20.0, 10.0]);
    assert!(p.edges.iter().all(|e| e.value == 1));
}

#[test]
fn polygons_refuse_bad_kinds() {
    assert!(polygons_err("polygons-bad-kind.geojson").contains("volume"));
    let e = polygons_err("polygons-bad-linestring-area.geojson");
    assert!(e.contains("tissue") && e.contains("LineString"), "{e}");
    let e = polygons_err("polygons-bad-tag-both.geojson");
    assert!(
        e.contains("tissue") && e.contains("area and as an edge"),
        "{e}"
    );
    let e = polygons_err("polygons-bad-value.geojson");
    assert!(e.contains("value"), "{e}");
}

#[test]
fn schema_carries_the_hole_radius_defaults() {
    let schema = contracts::schema_scene();
    let variants = schema["$defs"]["MeshSource"]["oneOf"].as_array().unwrap();
    let lobules = variants
        .iter()
        .find(|v| v["properties"].get("centre").is_some())
        .expect("lobules variant")["properties"]
        .clone();
    for (hole, default) in [("centre", 30.0), ("corner", 25.0)] {
        let node = &lobules[hole];
        let radius = &node["properties"]["radius_um"];
        assert_eq!(radius["default"], serde_json::json!(default), "{hole}");
        assert_eq!(radius["exclusiveMinimum"], serde_json::json!(0), "{hole}");
        assert_eq!(node["required"], serde_json::json!(["tag"]), "{hole}");
    }
}

fn result_meta(name: &str) -> Result<ResultMeta, String> {
    let text = std::fs::read_to_string(data(name)).unwrap();
    let meta: ResultMeta = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    meta.check().map_err(|e| e.to_string())?;
    Ok(meta)
}

#[test]
fn result_meta_round_trip() {
    for name in ["result-meta-minimal.json", "result-meta-raster.json"] {
        let meta = result_meta(name).unwrap();
        let original: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(data(name)).unwrap()).unwrap();
        assert_eq!(serde_json::to_value(&meta).unwrap(), original, "{name}");
    }
    let raster = result_meta("result-meta-raster.json")
        .unwrap()
        .raster
        .unwrap();
    assert_eq!(raster.levels, 3);
    assert_eq!(raster.shapes[1], [130, 150]);
}

#[test]
fn result_meta_rejects() {
    for (name, word) in [
        ("unknown-field", "extra"),
        ("version", "result/2"),
        ("written", "written"),
        ("complete", "complete"),
        ("times", "times"),
        ("duplicate-field", "glucose"),
        ("levels", "levels"),
        ("shapes", "shapes"),
        ("um-per-px", "um_per_px"),
        ("tag-dim", "dim"),
        ("duplicate-tag", "tissue"),
        ("duplicate-cell-value", "protein"),
        ("shape-zero", "shapes"),
    ] {
        let e = result_meta(&format!("result-meta-bad-{name}.json")).unwrap_err();
        assert!(e.contains(word), "{name}: {e}");
    }
}

#[test]
fn result_schema_rejects_what_it_can_express() {
    // Rules JSON Schema cannot state (uniqueness by key, cross-field rules) are only in `check`.
    for name in ["unknown-field", "version", "levels", "um-per-px", "tag-dim"] {
        let file = format!("result-meta-bad-{name}.json");
        assert!(!schema_accepts(contracts::schema_result(), &file), "{name}");
    }
    for file in ["result-meta-minimal.json", "result-meta-raster.json"] {
        assert!(schema_accepts(contracts::schema_result(), file), "{file}");
    }
}

#[test]
fn reserved_tag_names_are_refused() {
    for name in ["depth", "_boundary_", "_regions_", "pde_floating"] {
        assert!(contracts::RESERVED_TAGS.contains(&name), "{name}");
    }
    let read = |name: &str| -> serde_json::Value {
        serde_json::from_str(&std::fs::read_to_string(data(name)).unwrap()).unwrap()
    };
    for reserved in contracts::RESERVED_TAGS {
        let mut p = read("polygons-minimal.geojson");
        p["features"][0]["properties"]["tag"] = serde_json::json!(reserved);
        let e = Polygons::from_value(p, &data("polygons-minimal.geojson"))
            .unwrap_err()
            .to_string();
        assert!(e.contains(reserved) && e.contains("reserved"), "{e}");
        let mut s = read("scene-lobules.json");
        s["mesh"]["area_tag"] = serde_json::json!(reserved);
        let e = Scene::from_value(s, &data("scene-lobules.json"))
            .unwrap_err()
            .to_string();
        assert!(e.contains("area_tag") && e.contains("reserved"), "{e}");
    }
}
