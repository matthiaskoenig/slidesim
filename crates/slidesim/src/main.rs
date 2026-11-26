//! `slidesim`: spatial simulations of diffusing fields on tissue meshes.
//!
//! `slidesim run <study.json> --out <dir> [-- <petsc options>]` runs a `study/1` end to end (see
//! [`run`]), on one process or under `mpiexec`.
//!
//! Exit code 0 on success. Otherwise 1, with the error (and its causes) on stderr. On several
//! processes an error ends every process (`MPI_Abort`, exit code 1), since the others may wait
//! for the failing one in a collective call forever; each failing process prints its error
//! prefixed with its rank.

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
    about = "Spatial simulations of diffusing fields on tissue meshes"
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
}

const RUN_HELP: &str = "\
Output files in DIR:
  <field id>_<index>.vtu  the field at the output time number <index> (of time.outputs, from 0),
                          for ParaView: point data '<field id>.mM'
  summary.json            per field and output time: times (s), l2_norm (mM*um), total_mass
                          (mM*um^2, the integral of the field weighed by the porosity) and
                          mass_balance (mM*um^2/s, null at time 0): storage_rate, source,
                          advection_defect and, per boundary tag, the advective and diffusive
                          inflow (storage_rate = source + inflows - advection_defect); with
                          perfusion also perfusion.tag_flows (um^2/s per pressure tag,
                          positive into the domain); phases: the wall time (s) of process 0
                          in setup, pressure (the flow), transport and output
  pressure.vtu            with perfusion: the pressure, point data 'pressure.Pa'
  velocity.vtu            with perfusion: the Darcy velocity (um/s), the point vector 'velocity'
Files <field id>_<index>.vtu, pressure.vtu and velocity.vtu of an earlier run in DIR are removed
first; other files stay.

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
    match run::run(&petsc, &args.study, &args.out, started) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            let comm = petsc.comm();
            if comm.size() > 1 {
                eprintln!("[rank {}] error: {}", comm.rank(), causes(&e));
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
