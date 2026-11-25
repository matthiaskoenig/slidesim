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
    let version = match Command::new("mpiexec").arg("--version").output() {
        Ok(out) => String::from_utf8_lossy(&out.stdout).into_owned(),
        Err(e) if e.kind() == ErrorKind::NotFound => {
            eprintln!("skipped: no mpiexec");
            return;
        }
        Err(e) => panic!("cannot run mpiexec: {e}"),
    };
    let open_mpi = version.contains("Open MPI") || version.contains("OpenRTE");
    let exe = std::env::current_exe().unwrap();
    for test in tests {
        let mut cmd = Command::new("mpiexec");
        if open_mpi {
            // More processes than cores is fine for the small meshes of tests.
            cmd.arg("--oversubscribe");
        }
        cmd.args(["-n", &ranks.to_string()])
            .arg(&exe)
            .args(["--exact", test, "--test-threads", "1", "--color", "never"])
            .env(MPI_RANKS, ranks.to_string())
            .envs(env.iter().copied())
            // Its own process group, so that a timeout kills mpiexec and every process.
            .process_group(0)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let (status, stdout, stderr) = run_with_timeout(cmd, PARALLEL_TIMEOUT);
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

/// Runs `cmd` (in its own process group) for at most `timeout`: its exit status, or `None` if it
/// was killed (with its whole process group) for taking longer; and its stdout and stderr.
fn run_with_timeout(mut cmd: Command, timeout: Duration) -> (Option<ExitStatus>, String, String) {
    let mut child = cmd.spawn().unwrap();
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
            // The group id is the pid of mpiexec (`process_group(0)`).
            let group = format!("-{}", child.id());
            Command::new("kill")
                .args(["-KILL", "--", &group])
                .status()
                .unwrap();
            child.wait().unwrap();
            break None;
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    (status, stdout.join().unwrap(), stderr.join().unwrap())
}
