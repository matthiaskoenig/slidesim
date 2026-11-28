//! Support for slidesim's tests (a dev-dependency only).
//!
//! [`parallel`] runs selected tests of the current test binary again under `mpiexec`, for the
//! code paths a single process does not take. nextest runs each test in its own process, so each
//! run initializes PETSc (and MPI) once. A test reads the number of processes it runs on with
//! [`ranks`].
//!
//! Usage, in a test file:
//!
//! ```no_run
//! #[test]
//! fn parallel_on_2_processes() {
//!     test_support::parallel(2, &["mpi_world_has_the_expected_size", "some_test"]);
//! }
//! ```
//!
//! For other programs (a binary of the workspace, or a test expected to fail), [`Mpiexec`] makes
//! the `mpiexec` command and [`run_with_timeout`] runs it, a hang killed after a timeout.
//!
//! A test which starts `mpiexec` needs an entry in `.config/nextest.toml` (`threads-required`,
//! one test thread per process), and its timeouts stay below nextest's `slow-timeout` there, so
//! that they end a hanging `mpiexec` with all its processes.

use std::ffi::OsStr;
use std::io::{ErrorKind, Read};
use std::os::unix::process::CommandExt;
use std::process::{Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

/// Set in the runs [`parallel`] starts under `mpiexec`: the number of processes.
pub const MPI_RANKS: &str = "SLIDESIM_TEST_MPI_RANKS";

/// How long one run under `mpiexec` may take: a hang (a process waiting for another which left a
/// collective call with an error) fails the test.
pub const PARALLEL_TIMEOUT: Duration = Duration::from_secs(120);

/// The number of processes this test runs on: as started by [`parallel`], else 1.
pub fn ranks() -> usize {
    std::env::var(MPI_RANKS).map_or(1, |n| n.parse().expect("number of processes"))
}

/// `text`, a `summary.json`, without its wall times: `phases`, the last key, and the `seconds` of
/// the cells.
pub fn without_timings(text: &str) -> String {
    let at = text.find(",\n  \"phases\": {").expect("phases last");
    let mut kept = Vec::new();
    let mut lines = text[..at].lines();
    while let Some(line) = lines.next() {
        if line.trim_start().starts_with("\"seconds\": {") {
            // Up to the end of the object.
            lines.find(|l| l.trim_start().starts_with('}'));
            continue;
        }
        kept.push(line);
    }
    kept.join("\n")
}

/// Runs each of `tests` of the current test binary under `mpiexec -n ranks`, one run per test,
/// and panics unless every process of every run passes. Does nothing inside such a run, and is
/// skipped (with a message) if there is no `mpiexec`.
pub fn parallel(ranks: usize, tests: &[&str]) {
    parallel_with_env(ranks, tests, &[]);
}

/// As [`parallel`], with the environment variables `env` (name, value) set in the runs: for
/// example results of this process for the runs to compare theirs with.
pub fn parallel_with_env(ranks: usize, tests: &[&str], env: &[(&str, &str)]) {
    if std::env::var_os(MPI_RANKS).is_some() {
        // A run under mpiexec: do not start runs from it.
        return;
    }
    let Some(mpiexec) = Mpiexec::find() else {
        eprintln!("skipped: no mpiexec");
        return;
    };
    let exe = std::env::current_exe().unwrap();
    for test in tests {
        let mut cmd = mpiexec.command(ranks, &exe);
        cmd.args(["--exact", test, "--test-threads", "1", "--color", "never"])
            .env(MPI_RANKS, ranks.to_string())
            .envs(env.iter().copied());
        let Run {
            status,
            stdout,
            stderr,
        } = run_with_timeout(cmd, PARALLEL_TIMEOUT);
        let output = format!("{stdout}\n{stderr}");
        let status = status.unwrap_or_else(|| {
            panic!("{test} on {ranks} processes timed out after {PARALLEL_TIMEOUT:?}:\n{output}")
        });
        assert!(
            status.success(),
            "{test} on {ranks} processes failed ({status}):\n{output}"
        );
        // Each process ran the test (and not zero tests): output of the processes may be
        // interleaved, so count the summaries in the whole text.
        let ran = stdout.matches("1 passed; 0 failed").count();
        assert_eq!(
            ran, ranks,
            "{test} ran on {ran} of {ranks} processes:\n{output}"
        );
    }
}

/// The `mpiexec` of the system, for runs of any program on several processes.
#[derive(Debug, Clone, Copy)]
pub struct Mpiexec {
    /// Open MPI, which refuses more processes than cores unless told otherwise.
    open_mpi: bool,
}

impl Mpiexec {
    /// The `mpiexec` on the `PATH`, or `None` if there is none (the caller skips its test).
    /// Panics if it is there but cannot run.
    pub fn find() -> Option<Mpiexec> {
        match Command::new("mpiexec").arg("--version").output() {
            Ok(out) => {
                let version = String::from_utf8_lossy(&out.stdout);
                Some(Mpiexec {
                    open_mpi: version.contains("Open MPI") || version.contains("OpenRTE"),
                })
            }
            Err(e) if e.kind() == ErrorKind::NotFound => None,
            Err(e) => panic!("cannot run mpiexec: {e}"),
        }
    }

    /// `mpiexec -n ranks program`, in its own process group (so that [`run_with_timeout`] can
    /// kill mpiexec and every process). Arguments of `program` and the environment are for the
    /// caller to add.
    pub fn command(&self, ranks: usize, program: impl AsRef<OsStr>) -> Command {
        let mut cmd = Command::new("mpiexec");
        if self.open_mpi {
            // More processes than cores is fine for the small meshes of tests.
            cmd.arg("--oversubscribe");
        }
        cmd.args(["-n", &ranks.to_string()])
            .arg(program)
            .process_group(0);
        cmd
    }
}

/// The result of [`run_with_timeout`].
#[derive(Debug)]
pub struct Run {
    /// The exit status, or `None` if the run was killed for taking too long.
    pub status: Option<ExitStatus>,
    pub stdout: String,
    pub stderr: String,
}

/// Runs `cmd` for at most `timeout`, with its stdout and stderr piped and collected. If it takes
/// longer, it is killed: with its whole process group if `cmd` was made the leader of one
/// (`process_group(0)`, as [`Mpiexec::command`] does), else alone.
pub fn run_with_timeout(mut cmd: Command, timeout: Duration) -> Run {
    let mut child = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| panic!("cannot run {cmd:?}: {e}"));
    let read = |mut pipe: Box<dyn Read + Send>| {
        std::thread::spawn(move || {
            let mut text = String::new();
            pipe.read_to_string(&mut text).unwrap();
            text
        })
    };
    let stdout = read(Box::new(child.stdout.take().unwrap()));
    let stderr = read(Box::new(child.stderr.take().unwrap()));
    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break Some(status);
        }
        if start.elapsed() > timeout {
            // A leader's group id is its pid; killing a group which does not exist fails, so
            // the child alone is killed as well.
            let group = format!("-{}", child.id());
            let _ = Command::new("kill")
                .args(["-KILL", "--", &group])
                .stderr(Stdio::null())
                .status();
            let _ = child.kill();
            child.wait().unwrap();
            break None;
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    Run {
        status,
        stdout: stdout.join().unwrap(),
        stderr: stderr.join().unwrap(),
    }
}
