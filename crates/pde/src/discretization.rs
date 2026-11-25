//! The P1 discretization of a mesh level: `PetscFE`, section, and libCEED restrictions, bases
//! and quadrature geometry.

use std::collections::BTreeSet;
use std::rc::Rc;

use ceed::{Basis, CVector, Ceed, ElemRestriction, EvalMode, FieldVector, Operator, QFunction};
use mesh::Mesh;
use petsc::{AsVecMut, Dm, Fe, InsertMode, LocalOffsets, Matrix, ShellOperator, Vector, sys};

use crate::qfunctions::{self as qf, GEOMETRY_2D_SIZE, TRANSPORT_DATA_SIZE};
use crate::{PdeError, Result};

/// The P1 discretization of one level of a [`Mesh`], shared by the operators built on it.
///
/// It owns its own DM: a clone of the level ([`Dm::clone_mesh`]) with the P1 field and the
/// essential boundaries, so several discretizations of a level do not interfere.
///
/// Vectors: global vectors ([`Discretization::global_vector`]) hold the dofs a process owns,
/// without essential dofs; local vectors ([`Discretization::local_vector`]) hold every dof of
/// the local cells, ghosts and essential dofs included.
///
/// Values at quadrature points (see [`Discretization::quadrature_points`]) are stored cell by
/// cell, in the order of the local cells: point `q` of local cell `e` at `e * Q + q`, for `Q`
/// points per cell.
#[derive(Clone, Debug)]
pub struct Discretization {
    shared: Rc<Shared>,
}

/// What the operators of a discretization share.
#[derive(Debug)]
pub(crate) struct Shared {
    pub(crate) dm: Dm,
    pub(crate) ceed: Ceed,
    /// The P1 field: local vector to element nodes.
    pub(crate) restriction: ElemRestriction,
    pub(crate) basis: Basis,
    /// The quadrature geometry ([`GEOMETRY_2D_SIZE`] components per point).
    pub(crate) q_data: CVector,
    pub(crate) q_data_restriction: ElemRestriction,
    /// One value per quadrature point, in the layout of [`Discretization`].
    pub(crate) point_restriction: ElemRestriction,
    /// The point data of the transport operator ([`TRANSPORT_DATA_SIZE`] components per point).
    pub(crate) p_data_restriction: ElemRestriction,
    /// For each entry of a local vector, its global index (negative for essential dofs).
    pub(crate) local_to_global: Vec<i64>,
    /// The sizes of global vectors: on this process, and in all.
    pub(crate) global_sizes: (usize, usize),
    /// The length of local vectors.
    pub(crate) local_size: usize,
    points: Vec<[f64; 2]>,
    /// The weights of the points: `w det J`, in the layout of [`Discretization`].
    weights: Vec<f64>,
    /// The coordinates of the node of each entry of a local vector.
    nodes: Vec<[f64; 2]>,
    /// The essential tags and the local dofs they constrain, in the order given.
    essential: Vec<(String, Vec<usize>)>,
    /// The area tags of the mesh (its tags of the dimension of the mesh), sorted by name: the
    /// same on every process.
    pub(crate) area_tags: Vec<String>,
    /// For each local cell, the index in `area_tags` of its area tag, if any.
    cell_tags: Vec<Option<u32>>,
}

/// The quadrature degree of degree `degree` elements: mass matrices are integrated exactly.
fn quadrature_degree(degree: usize) -> usize {
    2 * degree
}

impl Discretization {
    /// The P`degree` discretization of level `level` of `mesh` (coarse first, see
    /// [`Mesh::levels`]) with natural boundary conditions everywhere. Collective.
    ///
    /// Errors: [`PdeError::Unsupported`] unless `degree` is 1 and the mesh is of triangles in
    /// 2D; [`PdeError::NoLevel`]; [`PdeError::InvertedElement`] if a cell has a non-positive
    /// Jacobian determinant (on every process).
    pub fn new(mesh: &Mesh, level: usize, ceed: &Ceed, degree: u32) -> Result<Discretization> {
        Discretization::with_essential(mesh, level, ceed, degree, &[])
    }

    /// As [`Discretization::new`], with essential (Dirichlet) boundaries on the tags `essential`:
    /// the dofs on the points they mark, and on their closure, are constrained and left out of
    /// global vectors. Operators see zero values there, unless set in a local vector
    /// ([`Discretization::set_essential_values`]). Errors also for a tag the mesh does not have
    /// ([`PdeError::Mesh`]).
    pub fn with_essential(
        mesh: &Mesh,
        level: usize,
        ceed: &Ceed,
        degree: u32,
        essential: &[&str],
    ) -> Result<Discretization> {
        Discretization::build(mesh, level, ceed, degree, essential, None)
    }

    /// The discretizations of every level of `mesh`, coarse first, each as
    /// [`Discretization::with_essential`], and each the next coarser level of the one after it
    /// ([`Dm::set_coarse_dm`] of their DMs): the hierarchy of geometric multigrid
    /// ([`crate::configure_multigrid`]). Collective.
    pub fn hierarchy(
        mesh: &Mesh,
        ceed: &Ceed,
        degree: u32,
        essential: &[&str],
    ) -> Result<Vec<Discretization>> {
        let mut levels: Vec<Discretization> = Vec::with_capacity(mesh.levels().len());
        for level in 0..mesh.levels().len() {
            let coarse = levels.last().map(Discretization::dm);
            let disc = Discretization::build(mesh, level, ceed, degree, essential, coarse)?;
            levels.push(disc);
        }
        Ok(levels)
    }

    /// The discretization of [`Discretization::with_essential`], with the DM `coarse` as the
    /// next coarser level of its DM, if given.
    fn build(
        mesh: &Mesh,
        level: usize,
        ceed: &Ceed,
        degree: u32,
        essential: &[&str],
        coarse: Option<&Dm>,
    ) -> Result<Discretization> {
        if degree != 1 {
            return Err(PdeError::Unsupported(format!(
                "elements of degree {degree} (only degree 1 is supported)"
            )));
        }
        let levels = mesh.levels();
        let source = levels.get(level).ok_or(PdeError::NoLevel {
            level,
            levels: levels.len(),
        })?;
        let dim = source.dimension()?;
        if dim != 2 {
            return Err(PdeError::Unsupported(format!(
                "meshes of dimension {dim} (only dimension 2 is supported)"
            )));
        }
        // A process without cells has no cell type: the processes decide together.
        let other_cells = !source.height_stratum(0)?.is_empty() && !source.is_simplex()?;
        if source.comm().max_int(other_cells.into())? > 0 {
            return Err(PdeError::Unsupported(
                "cells other than triangles".to_string(),
            ));
        }
        let tags = essential
            .iter()
            .map(|name| mesh.tag(name))
            .collect::<Result<Vec<_>, _>>()?;
        let degree = degree as usize;

        let mut dm = source.clone_mesh()?;
        if let Some(coarse) = coarse {
            dm.set_coarse_dm(coarse)?;
        }
        // The affine coordinate element, for the restriction of the coordinates.
        dm.set_coordinate_disc(None, false)?;
        let fe = Fe::lagrange(dim, 1, true, degree, quadrature_degree(degree))?;
        dm.add_field(&fe)?;
        dm.create_ds()?;
        for tag in &tags {
            let values = dm.label_values_global(&tag.name)?;
            dm.add_essential_boundary(&tag.name, &tag.name, &values, 0)?;
        }
        // The first vector creates the section (collective).
        let global = Vector::create_global(&dm)?;
        let global_sizes = (global.local_size()?, global.size()?);
        let local_size = Vector::create_local(&dm)?.local_size()?;
        let local_to_global = dm
            .local_to_global_indices()?
            .into_iter()
            .map(i64::from)
            .collect();
        let essential = tags
            .iter()
            .map(|tag| Ok((tag.name.clone(), constrained_dofs(&dm, &tag.name)?)))
            .collect::<Result<Vec<_>>>()?;

        let offsets = dm.local_offsets(0)?;
        let (area_tags, cell_tags) = area_tags(mesh, &dm, dim)?;
        let x_offsets = dm.coordinate_dm()?.local_offsets(0)?;
        let coordinates = dm.local_coordinates()?;
        let num_cells = offsets.num_cells;
        if offsets.local_size != local_size
            || cell_tags.len() != num_cells
            || x_offsets.num_cells != num_cells
            || x_offsets.num_comp != dim
            || x_offsets.local_size != coordinates.len()
        {
            return Err(PdeError::Unsupported(format!(
                "a cell restriction of {} cells ({} cells in the mesh) for local vectors of length \
                 {local_size}, and of {} cells with {} components for {} coordinates",
                num_cells,
                cell_tags.len(),
                x_offsets.num_cells,
                x_offsets.num_comp,
                coordinates.len()
            )));
        }

        // The node of an entry of a local vector: a vertex, where the coordinate restriction
        // has the coordinates of the same node of the same cell.
        let mut nodes = vec![[f64::NAN; 2]; local_size];
        for (&dof, &x) in offsets.offsets.iter().zip(&x_offsets.offsets) {
            let (dof, x) = (dof as usize, x as usize);
            nodes[dof] = [coordinates[x], coordinates[x + 1]];
        }

        let basis = basis_of(ceed, &fe, 1)?;
        let basis_x = basis_of(ceed, &fe, dim)?;
        let q = basis.num_quadrature_points();
        let restriction = restriction_of(ceed, &offsets, 1)?;
        let restriction_x = restriction_of(ceed, &x_offsets, dim)?;
        let x = CVector::from_slice(ceed, &coordinates)?;
        let per_point = |num_comp: usize| {
            ElemRestriction::strided_with(
                ceed,
                num_cells,
                q,
                num_comp,
                num_cells * q * num_comp,
                [1, q, q * num_comp],
            )
        };
        let q_data_restriction = per_point(GEOMETRY_2D_SIZE)?;
        let point_restriction = per_point(1)?;
        let p_data_restriction = per_point(TRANSPORT_DATA_SIZE)?;

        // The geometry: w det(J) and J^-1 at every point.
        // SAFETY: the fields match what SetupGeometry2D reads and writes; no context.
        let mut qf_geometry = unsafe {
            QFunction::interior(
                ceed,
                1,
                Some(qf::slidesim_pde_SetupGeometry2D),
                qf::SETUP_GEOMETRY_2D,
            )
        }?;
        qf_geometry.add_input("dx", dim * dim, EvalMode::Grad)?;
        qf_geometry.add_input("weight", 1, EvalMode::Weight)?;
        qf_geometry.add_output("q_data", GEOMETRY_2D_SIZE, EvalMode::None)?;
        let mut op = Operator::new(ceed, &qf_geometry)?;
        op.set_field(
            "dx",
            Some(&restriction_x),
            Some(&basis_x),
            FieldVector::Active,
        )?;
        op.set_field("weight", None, Some(&basis_x), FieldVector::None)?;
        op.set_field(
            "q_data",
            Some(&q_data_restriction),
            None,
            FieldVector::Active,
        )?;
        let mut q_data = CVector::new(ceed, q_data_restriction.l_size())?;
        op.apply(&x, &mut q_data)?;
        check_orientation(&dm, &q_data, &basis.quadrature_weights()?, num_cells)?;
        // w det(J) at [e][0][q].
        let weights = q_data.with_array(|values| {
            values
                .chunks_exact(GEOMETRY_2D_SIZE * q)
                .flat_map(|cell| cell[..q].iter().copied())
                .collect()
        })?;

        // The coordinates of the points.
        // SAFETY: the fields match what QuadraturePoints2D reads and writes; no context.
        let mut qf_points = unsafe {
            QFunction::interior(
                ceed,
                1,
                Some(qf::slidesim_pde_QuadraturePoints2D),
                qf::QUADRATURE_POINTS_2D,
            )
        }?;
        qf_points.add_input("x", dim, EvalMode::Interp)?;
        qf_points.add_output("x_q", dim, EvalMode::None)?;
        let mut op = Operator::new(ceed, &qf_points)?;
        op.set_field(
            "x",
            Some(&restriction_x),
            Some(&basis_x),
            FieldVector::Active,
        )?;
        let points_restriction = per_point(dim)?;
        op.set_field("x_q", Some(&points_restriction), None, FieldVector::Active)?;
        let mut x_q = vec![0.0; points_restriction.l_size()];
        op.apply_slices(&coordinates, &mut x_q)?;
        // Cell e holds x then y of its points: [e][c][q].
        let points = x_q
            .chunks_exact(dim * q)
            .flat_map(|cell| (0..q).map(move |i| [cell[i], cell[q + i]]))
            .collect();

        Ok(Discretization {
            shared: Rc::new(Shared {
                dm,
                ceed: ceed.clone(),
                restriction,
                basis,
                q_data,
                q_data_restriction,
                point_restriction,
                p_data_restriction,
                local_to_global,
                global_sizes,
                local_size,
                points,
                weights,
                nodes,
                essential,
                area_tags,
                cell_tags,
            }),
        })
    }

    pub(crate) fn from_shared(shared: Rc<Shared>) -> Discretization {
        Discretization { shared }
    }

    pub(crate) fn shared(&self) -> &Rc<Shared> {
        &self.shared
    }

    /// The DM of the discretization: the mesh level with the P1 field and the essential
    /// boundaries.
    pub fn dm(&self) -> &Dm {
        &self.shared.dm
    }

    /// The libCEED context of the operators.
    pub fn ceed(&self) -> &Ceed {
        &self.shared.ceed
    }

    /// The physical coordinates of the quadrature points of the local cells, cell by cell (see
    /// [`Discretization`]).
    pub fn quadrature_points(&self) -> &[[f64; 2]] {
        &self.shared.points
    }

    /// The weights of the quadrature points of the local cells, in the layout of
    /// [`Discretization::quadrature_points`]: the quadrature weight times the Jacobian
    /// determinant, so that `sum_q weights[q] f(x_q)` is the integral of `f` over the local cells.
    pub fn quadrature_weights(&self) -> &[f64] {
        &self.shared.weights
    }

    /// The number of quadrature points of each cell.
    pub(crate) fn points_per_cell(&self) -> usize {
        self.shared.basis.num_quadrature_points()
    }

    /// The name of the area tag of local cell `cell` (in the order of the local cells, see
    /// [`Discretization`]): of the tags of the dimension of the mesh (2), the one which marks the
    /// cell, the first by name if several do. `None` if no area tag marks it, or there is no such
    /// local cell.
    pub fn cell_area_tag(&self, cell: usize) -> Option<String> {
        self.cell_area_tag_index(cell)
            .map(|i| self.shared.area_tags[i].clone())
    }

    /// The index of the area tag of local cell `cell` in the area tags of the discretization,
    /// which are the same on every process (see [`Discretization::cell_area_tag`]).
    pub(crate) fn cell_area_tag_index(&self, cell: usize) -> Option<usize> {
        let index = (*self.shared.cell_tags.get(cell)?)?;
        Some(index as usize)
    }

    /// The coordinates of the nodes of the entries of local vectors: of the vertices, for P1.
    pub fn node_coordinates(&self) -> &[[f64; 2]] {
        &self.shared.nodes
    }

    /// Sets the global vector `u` to the interpolant of `f`, the values of `f` at the nodes of
    /// its dofs (essential dofs are not in `u`). Collective.
    pub fn interpolate(&self, f: impl Fn([f64; 2]) -> f64, u: &mut Vector) -> Result<()> {
        let mut local = self.local_vector()?;
        local.with_array_mut(|values| {
            for (v, &x) in values.iter_mut().zip(&self.shared.nodes) {
                *v = f(x);
            }
        })?;
        self.shared
            .dm
            .local_to_global(&local, InsertMode::Insert, u)?;
        Ok(())
    }

    /// The load of inward boundary fluxes, `sum_i int_(Gamma_i) q_i v` for the test functions
    /// `v` of the global dofs, as a global vector: `fluxes` are pairs of a tag of edges
    /// `Gamma_i` and a flux `q_i` (in mM*um/s, uniform on the tag). Each edge counts once, on the
    /// process which owns it; the vertices on the tags are ignored. Collective.
    ///
    /// Errors (on every process): [`PdeError::Unsupported`] if a tag marks cells (a region,
    /// whose refinements mark interior edges too).
    pub fn boundary_load(&self, fluxes: &[(&str, f64)]) -> Result<Vector> {
        let dm = &self.shared.dm;
        let cells = dm.height_stratum(0)?;
        let edges = dm.depth_stratum(1)?;
        let vertices = dm.depth_stratum(0)?;
        let ghosts = dm.ghost_points()?;
        let mut tagged = Vec::with_capacity(fluxes.len());
        for &(tag, _) in fluxes {
            let mut points = BTreeSet::new();
            for value in dm.label_values(tag)? {
                points.extend(dm.stratum_points(tag, value)?);
            }
            let marks_cells = points.iter().any(|p| cells.contains(p));
            if dm.comm().max_int(marks_cells.into())? > 0 {
                return Err(PdeError::Unsupported(format!(
                    "a boundary flux on tag '{tag}', which marks cells"
                )));
            }
            tagged.push(points);
        }
        // P1 on an edge of length l: q l / 2 to each of its two vertices.
        let mut contributions = Vec::new();
        for (&(tag, q), points) in fluxes.iter().zip(tagged) {
            for edge in points {
                if !edges.contains(&edge) || ghosts.binary_search(&edge).is_ok() {
                    continue;
                }
                let mut ends = Vec::with_capacity(2);
                for p in dm.closure(edge)? {
                    if vertices.contains(&p) {
                        ends.push(dm.local_dofs(p)?.start);
                    }
                }
                let &[a, b] = ends.as_slice() else {
                    return Err(PdeError::Unsupported(format!(
                        "edge {edge} of tag '{tag}' with {} vertices",
                        ends.len()
                    )));
                };
                let ([xa, ya], [xb, yb]) = (self.shared.nodes[a], self.shared.nodes[b]);
                let half = 0.5 * q * (xb - xa).hypot(yb - ya);
                contributions.extend([(a, half), (b, half)]);
            }
        }
        let mut local = self.local_vector()?;
        local.with_array_mut(|values| {
            values.fill(0.0);
            for (i, v) in contributions {
                values[i] += v;
            }
        })?;
        let mut load = self.global_vector()?;
        load.zero()?;
        dm.local_to_global(&local, InsertMode::Add, &mut load)?;
        Ok(load)
    }

    /// The number of quadrature points of the local cells.
    pub fn num_quadrature_points(&self) -> usize {
        self.shared.points.len()
    }

    /// A new global vector (`DMCreateGlobalVector`), without essential dofs. Collective.
    pub fn global_vector(&self) -> Result<Vector> {
        Ok(Vector::create_global(&self.shared.dm)?)
    }

    /// A new local vector (`DMCreateLocalVector`), with ghost and essential dofs.
    pub fn local_vector(&self) -> Result<Vector> {
        Ok(Vector::create_local(&self.shared.dm)?)
    }

    /// A shell matrix on global vectors of this discretization, applied by `op`. Collective.
    pub fn shell_matrix(&self, op: impl ShellOperator + 'static) -> Result<Matrix> {
        let (local, global) = self.shared.global_sizes;
        Ok(Matrix::shell(
            &self.shared.dm.comm(),
            local,
            global,
            Box::new(op),
        )?)
    }

    /// Sets the essential dofs of the tag `tag` in the local vector `local` to `value` (a
    /// Dirichlet value; the dofs of the closure of a point shared by several essential tags take
    /// the value of the tag set last). An error if `tag` is not one of the essential tags.
    pub fn set_essential_values(&self, local: &mut Vector, tag: &str, value: f64) -> Result<()> {
        let essential = &self.shared.essential;
        let Some((_, dofs)) = essential.iter().find(|(name, _)| name == tag) else {
            return Err(PdeError::NotEssential {
                tag: tag.to_string(),
                essential: essential.iter().map(|(name, _)| name.clone()).collect(),
            });
        };
        let length = local.local_size()?;
        if length != self.shared.local_size {
            return Err(PdeError::Petsc(petsc::Error {
                code: sys::PETSC_ERR_ARG_SIZ,
                message: format!(
                    "a vector of length {length} is not a local vector (length {})",
                    self.shared.local_size
                ),
            }));
        }
        local.with_array_mut(|values| {
            for &i in dofs {
                values[i] = value;
            }
        })?;
        Ok(())
    }
}

/// The area tags of `mesh` (of dimension `dim`, sorted by name), and for each local cell of `dm`
/// (a level of `mesh`, with its labels) the index of its area tag among them: of several, the
/// first.
fn area_tags(mesh: &Mesh, dm: &Dm, dim: usize) -> Result<(Vec<String>, Vec<Option<u32>>)> {
    let names: Vec<String> = mesh
        .tags()
        .iter()
        .filter(|tag| usize::from(tag.dim) == dim)
        .map(|tag| tag.name.clone())
        .collect();
    let cells = dm.height_stratum(0)?;
    let mut tags = vec![None; cells.len()];
    // The last written wins: the first by name.
    for (index, name) in names.iter().enumerate().rev() {
        let index = u32::try_from(index)
            .map_err(|_| PdeError::Unsupported(format!("{} area tags", names.len())))?;
        for value in dm.label_values(name)? {
            for point in dm.stratum_points(name, value)? {
                if cells.contains(&point) {
                    tags[(point - cells.start) as usize] = Some(index);
                }
            }
        }
    }
    Ok((names, tags))
}

/// The local dofs on the points the label `name` marks, and on their closure, sorted.
fn constrained_dofs(dm: &Dm, name: &str) -> Result<Vec<usize>> {
    let mut dofs = BTreeSet::new();
    for value in dm.label_values(name)? {
        for point in dm.stratum_points(name, value)? {
            for p in dm.closure(point)? {
                dofs.extend(dm.local_dofs(p)?);
            }
        }
    }
    Ok(dofs.into_iter().collect())
}

/// The libCEED H1 basis of the scalar element `fe` with `num_comp` components: the `PetscFE`
/// tabulation on PETSc's reference cell, with its nodes in closure order (the order of
/// `DMPlexGetLocalOffsets`).
fn basis_of(ceed: &Ceed, fe: &Fe, num_comp: usize) -> Result<Basis> {
    let quadrature = fe.quadrature()?;
    let t = fe.tabulation()?;
    let (q, p, dim) = (t.num_points, t.num_basis, t.dim);
    if t.num_comp != 1 || dim != quadrature.dim {
        return Err(PdeError::Unsupported(format!(
            "a basis of {} components in {dim}D on a quadrature in {}D",
            t.num_comp, quadrature.dim
        )));
    }
    // PETSc: values [q][p], derivatives [q][p][d], points [q][d]. libCEED: interp [q][p],
    // grad [d][q][p], q_ref [d][q].
    let mut grad = vec![0.0; dim * q * p];
    let mut q_ref = vec![0.0; dim * q];
    for d in 0..dim {
        for i in 0..q {
            for b in 0..p {
                grad[(d * q + i) * p + b] = t.derivatives[(i * p + b) * dim + d];
            }
            q_ref[d * q + i] = quadrature.points[i * dim + d];
        }
    }
    let topology = ceed::Topology::Triangle;
    Ok(Basis::h1(
        ceed,
        topology,
        num_comp,
        p,
        q,
        &t.values,
        &grad,
        &q_ref,
        &quadrature.weights,
    )?)
}

/// The libCEED restriction of the cell offsets `offsets`, with `num_comp` components
/// interlaced.
fn restriction_of(ceed: &Ceed, offsets: &LocalOffsets, num_comp: usize) -> Result<ElemRestriction> {
    // PetscInt is 32 or 64 bits, depending on the PETSc build; CeedInt is 32 bits.
    #[allow(clippy::useless_conversion)]
    let indices = offsets
        .offsets
        .iter()
        .map(|&o| ceed::sys::CeedInt::try_from(o))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| PdeError::Unsupported("local vectors beyond 2^31 entries".to_string()))?;
    Ok(ElemRestriction::new(
        ceed,
        offsets.num_cells,
        offsets.cell_size,
        num_comp,
        1,
        offsets.local_size,
        &indices,
    )?)
}

/// [`PdeError::InvertedElement`] on every process if a cell has a non-positive Jacobian
/// determinant at a quadrature point: the cell of the smallest global number among them. Each
/// process checks the cells it owns. `q_data` holds the geometry of the `num_cells` local cells,
/// `weights` are the quadrature weights. Collective.
fn check_orientation(dm: &Dm, q_data: &CVector, weights: &[f64], num_cells: usize) -> Result<()> {
    let q = weights.len();
    let numbering = dm.cell_numbering()?;
    if numbering.len() != num_cells {
        return Err(PdeError::Unsupported(format!(
            "{} numbered cells for {num_cells} cells of the restriction",
            numbering.len()
        )));
    }
    // The first non-positive w det(J) of each cell, at [e][0][q].
    let local = q_data.with_array(|values| {
        values
            .chunks_exact(GEOMETRY_2D_SIZE * q)
            .zip(&numbering)
            // A cell owned by another process is numbered -(g + 1).
            .filter(|&(_, &number)| number >= 0)
            .filter_map(|(cell, &number)| {
                let i = cell[..q]
                    .iter()
                    .position(|&w_det| w_det <= 0.0 || w_det.is_nan())?;
                Some((number, cell[i] / weights[i]))
            })
            .min_by_key(|&(global, _)| global)
    })?;
    let comm = dm.comm();
    // The smallest global number of an inverted cell, as the maximum of its negation.
    let key = comm.max_int(local.map_or(sys::PetscInt::MIN, |(cell, _)| -cell))?;
    if key == sys::PetscInt::MIN {
        return Ok(());
    }
    let cell = -key;
    let det = match local {
        Some((c, det)) if c == cell => det,
        _ => 0.0,
    };
    Err(PdeError::InvertedElement {
        cell: i64::from(cell),
        det: comm.sum_real(det)?,
    })
}
