//! `slidesim`: spatial simulations of diffusing fields on tissue meshes.
//!
//! `slidesim run <study.json> --out <dir> [-- <petsc options>]` runs a `study/1` end to end (see
//! [`run`], over the library `simulation`), on one process or under `mpiexec`. `slidesim mesh
//! <scene.json> --out <dir>` writes the mesh of a `scene/1` and its quality ([`mesh_command`]),
//! on one process.
//!
//! Exit code 0 on success. Otherwise 1, with the error (and its causes) on stderr. On several
//! processes an error ends every process (`MPI_Abort`, exit code 1), since the others may wait
//! for the failing one in a collective call forever; each failing process prints its error
//! prefixed with its rank. So does a panic (a bug): `[rank <k>] panic: <message> (at <location>)`.
//! An error which every process meets at the same point ([`simulation::OnEveryProcess`], as an
//! error of the mesher on process 0, which reaches the others) ends each process alone: each
//! prints it prefixed with its rank and exits with 1, without `MPI_Abort`.

mod mesh_command;
mod run;

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use clap::{Args, Parser, Subcommand};
use petsc::Petsc;

#[derive(Debug, Parser)]
#[command(
    name = "slidesim",
    version,
    about = "Spatial simulations of diffusing fields and cell models on tissue meshes"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run a study/1 and write its results.
    #[command(
        override_usage = "slidesim run <STUDY> --out <DIR> [-- <PETSC OPTIONS>...]",
        after_help = RUN_HELP
    )]
    Run(RunArgs),
    /// Write the mesh of a scene/1 and its quality.
    #[command(
        override_usage = "slidesim mesh <SCENE> --out <DIR>",
        after_help = MESH_HELP
    )]
    Mesh(MeshArgs),
}

const MESH_HELP: &str = "\
Output files in DIR:
  mesh.json     the mesh: vertices (um), triangles, the index of the area tag of each
                triangle, the tagged edges with the index of their tag, and the tags (name,
                value, dim: 2 for areas, 1 for edges)
  mesh.vtu      for ParaView: one cell array per area tag, named after it, with the value of
                the tag in its triangles and NaN elsewhere
  quality.json  the numbers of triangles and vertices, the smallest angle (min_angle_deg), the
                smallest and mean SICN, and for a generated mesh the smallest and largest ratio
                of an edge length to the size field (edge_ratio)
The quality is also printed in one line.

A gmsh mesh is read and converted; tagged polygons and lobules are meshed (the mesh a run
makes of the scene, before the refinements of the study). Runs on one process.";

#[derive(Debug, Args)]
struct MeshArgs {
    /// The scene/1 file.
    scene: PathBuf,
    /// The output directory, created if missing. mesh.json, mesh.vtu and quality.json in it are
    /// overwritten.
    #[arg(long, value_name = "DIR")]
    out: PathBuf,
}

const RUN_HELP: &str = "\
Output files in DIR:
  <field id>_<index>.vtu  the field at the output time number <index> (of time.outputs, from 0),
                          for ParaView: point data '<field id>.mM'
  cells_<index>.vtu       with cells: the cells at the output time, cell data of the coarse
                          elements named after the values: the states of the model (species
                          as concentrations), cells.outputs and the parameters of the element
  summary.json            per field and output time: times (s), l2_norm (mM*um), total_mass
                          (mM*um^2, the integral of the field weighed by the porosity) and
                          mass_balance (mM*um^2/s, over the interval since the output time
                          before, each term its mean rate there; null at time 0, and
                          throughout if PETSc options chose another time integration than
                          BDF2):
                          storage_rate, source, advection_defect, exchange (given by the cells)
                          and, per boundary tag, the advective and diffusive inflow and the
                          resupply of Dirichlet values next to the cells (storage_rate = source
                          + inflows + exchange - advection_defect); with perfusion also
                          perfusion.tag_flows (um^2/s per pressure tag, positive into the
                          domain) and perfusion.pressure_islands (the count and area_um2 of the
                          parts no pressure tag reaches: pressure 0, no flow); with cells also
                          cells: their count, steps and rejected steps, seconds (the time of
                          each process in its cells: mean and max over the processes), per
                          output time the amount each linked field got from them since t = 0
                          (exchanged) and the amounts of the species of the cell compartment
                          (amounts, mM*um^2); phases: the wall time (s) of process 0 in setup,
                          pressure (the flow), transport, exchange, cells and output
  pressure.vtu            with perfusion: the pressure, point data 'pressure.Pa'
  velocity.vtu            with perfusion: the Darcy velocity (um/s), the point vector 'velocity'
  result.zarr             the store result/1 for viewers (Zarr v3, zstd): the finest mesh, and
                          per output time every field at its vertices and the values of the
                          cells; written by process 0 as the output times are reached (its
                          attributes 'written' and 'complete' tell how far)
Files <field id>_<index>.vtu, cells_<index>.vtu, pressure.vtu and velocity.vtu of an earlier run
in DIR are removed first, result.zarr whole; other files stay.

The cell model of a study with cells is compiled with $CC (or cc) at the start of the run, and
cached in $SBMLSIM_C_CACHE (default: $XDG_CACHE_HOME/sbmlsim/c or ~/.cache/sbmlsim/c).

PETSc options after '--' go to PETSc's options database. The solvers of a field take the
prefix '<field id>_', for example '-oxygen_ksp_monitor' (residuals of the linear solves of the
field 'oxygen'), '-oxygen_ts_monitor' (its time steps) or '-oxygen_ksp_rtol 1e-8'. Precedence,
lowest first: slidesim's defaults ('-ksp_rtol 1e-10', solvers which do not converge are
errors), solver.petsc_options of the study, the options after '--'.";

#[derive(Debug, Args)]
struct RunArgs {
    /// The study/1 file.
    study: PathBuf,
    /// The output directory, created if missing. Outputs of an earlier run in it are removed or
    /// overwritten.
    #[arg(long, value_name = "DIR")]
    out: PathBuf,
    /// PETSc options, for example -oxygen_ksp_monitor.
    #[arg(last = true, value_name = "PETSC OPTIONS")]
    petsc_options: Vec<String>,
}

fn main() -> ExitCode {
    // The start of the run, for the phases of its summary.
    let started = Instant::now();
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(e) => {
            // Help and version go to stdout and succeed; usage errors go to stderr.
            let _ = e.print();
            return if e.use_stderr() {
                ExitCode::FAILURE
            } else {
                ExitCode::SUCCESS
            };
        }
    };
    match cli.command {
        Command::Run(args) => run_command(&args, started),
        Command::Mesh(args) => mesh(&args),
    }
}

fn mesh(args: &MeshArgs) -> ExitCode {
    let petsc = match Petsc::init(&[]) {
        Ok(petsc) => petsc,
        Err(e) => {
            eprintln!("error: cannot initialize PETSc: {e}");
            return ExitCode::FAILURE;
        }
    };
    let comm = petsc.comm();
    if comm.size() > 1 {
        // Each process finds this alone, and ends.
        if comm.rank() == 0 {
            eprintln!(
                "error: slidesim mesh runs on one process, not on {}",
                comm.size()
            );
        }
        return ExitCode::FAILURE;
    }
    match mesh_command::mesh(&petsc, &args.scene, &args.out) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {}", causes(&e));
            ExitCode::FAILURE
        }
    }
}

fn run_command(args: &RunArgs, started: Instant) -> ExitCode {
    let options: Vec<&str> = args.petsc_options.iter().map(String::as_str).collect();
    let petsc = match Petsc::init(&options) {
        Ok(petsc) => petsc,
        Err(e) => {
            eprintln!("error: cannot initialize PETSc: {e}");
            return ExitCode::FAILURE;
        }
    };
    if petsc.comm().size() > 1 {
        // A panic is a bug, but it must not leave the other processes waiting forever.
        petsc.abort_world_on_panic(1);
    }
    match run::run(&petsc, &args.study, &args.out, started) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            let comm = petsc.comm();
            if comm.size() > 1 {
                eprintln!("[rank {}] error: {}", comm.rank(), causes(&e));
                if e.downcast_ref::<simulation::OnEveryProcess>().is_some() {
                    // Every process has it: none waits for another.
                    return ExitCode::FAILURE;
                }
                petsc.abort_world(1);
            }
            eprintln!("error: {}", causes(&e));
            ExitCode::FAILURE
        }
    }
}

/// The error and its causes, `error: cause: cause`, without a cause whose text the ones before
/// already show (errors which print their source in their own message).
fn causes(e: &anyhow::Error) -> String {
    let mut text = String::new();
    for cause in e.chain() {
        let cause = cause.to_string();
        if text.contains(&cause) {
            continue;
        }
        if !text.is_empty() {
            text.push_str(": ");
        }
        text.push_str(&cause);
    }
    text
}
