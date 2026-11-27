//! Regenerate `schemas/*.schema.json`: `cargo run -p contracts --example write-schemas`.
use std::path::Path;

fn main() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../schemas");
    std::fs::create_dir_all(&dir).expect("create schemas dir");
    for (name, schema) in [
        ("scene-1.schema.json", contracts::schema_scene()),
        ("polygons-1.schema.json", contracts::schema_polygons()),
        ("study-1.schema.json", contracts::schema_study()),
    ] {
        let text = serde_json::to_string_pretty(&schema).expect("serialize") + "\n";
        std::fs::write(dir.join(name), text).expect("write schema");
    }
}
