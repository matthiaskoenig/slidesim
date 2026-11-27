//! Names a tag cannot have: the DMPlex labels PETSc and slidesim create themselves.

/// The labels PETSc creates on a DMPlex (topology, gmsh's generic sets, boundary markers, ghosts,
/// output, Mmg's boundary) and those slidesim adds to its meshes. A tag becomes a label of its
/// name, so a tag of one of these names would write into PETSc's own label, or be skipped when
/// the labels are read back.
pub const RESERVED_TAGS: [&str; 11] = [
    "depth",
    "celltype",
    "Cell Sets",
    "Face Sets",
    "Edge Sets",
    "Vertex Sets",
    "marker",
    "ghost",
    "vtk",
    "_boundary_",
    "pde_floating",
];

/// An error message unless `tag` is a name a tag may have: not empty, not reserved.
pub fn check_tag_name(tag: &str) -> Result<(), String> {
    if tag.is_empty() {
        Err("a tag must not be empty".to_string())
    } else if RESERVED_TAGS.contains(&tag) {
        Err(format!(
            "the tag '{tag}' is reserved (a label PETSc or slidesim creates itself)"
        ))
    } else {
        Ok(())
    }
}
