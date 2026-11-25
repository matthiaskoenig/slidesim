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
  summary.json            per field: times (s), l2_norm (mM*um), total_mass (mM*um^2)

PETSc options after '--' go to PETSc's options database. The solvers of a field take the
prefix '<field id>_', for example '-oxygen_ksp_monitor' (residuals of the linear solves of the
field 'oxygen') or '-oxygen_ts_monitor' (its time steps). Options in solver.petsc_options of
the study take precedence.";

#[derive(Debug, Args)]
struct RunArgs {
    /// The study/1 file.
    study: PathBuf,
    /// The output directory, created if missing; files in it are overwritten.
    #[arg(long, value_name = "DIR")]
    out: PathBuf,
    /// PETSc options, for example -oxygen_ksp_monitor.
    #[arg(last = true, value_name = "PETSC OPTIONS")]
    petsc_options: Vec<String>,
}

fn main() -> ExitCode {
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
        Command::Run(args) => run_command(&args),
    }
}

fn run_command(args: &RunArgs) -> ExitCode {
    let options: Vec<&str> = args.petsc_options.iter().map(String::as_str).collect();
    let petsc = match Petsc::init(&options) {
        Ok(petsc) => petsc,
        Err(e) => {
            eprintln!("error: cannot initialize PETSc: {e}");
            return ExitCode::FAILURE;
        }
    };
    match run::run(&petsc, &args.study, &args.out) {
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
