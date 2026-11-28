# shellcheck shell=bash
# Process hygiene for the scripts which run simulations: no process they start outlives them.
#
# Source it at the top of a script, before anything else runs, without arguments (it re-executes
# the script with the script's own arguments):
#
#   source "$(dirname "${BASH_SOURCE[0]}")/../lib/proc.sh"
#
# 1. The script runs in a process group (and session) of its own: unless it leads a session, it
#    re-executes itself in a child under `setsid`, and the process the caller started stays as
#    a relay which waits for it and returns its status. The relay gets TERM when the caller ends
#    (a wrapper killed, even with SIGKILL), keeps the caller's group (Ctrl-C of a terminal
#    reaches it), and the script gets TERM when the relay ends, however it ends.
# 2. On EXIT, INT, TERM and HUP it sends TERM to every other process of its group and to every
#    descendant (timeout and mpiexec make groups of their own), waits up to PROC_GRACE seconds
#    (default 5) for them to end, then sends KILL to those left.
# 3. `run_sim <limit> <cmd...>` runs a simulation with a wall-clock limit (a `timeout` duration,
#    e.g. 5m), confined to the CPUs PROC_CPUS (default 0-9) at nice 10:
#      setpriv --pdeathsig TERM timeout --kill-after=10s <limit> \
#          setpriv --pdeathsig TERM taskset -c <cpus> nice -n 10 <cmd...>
#    Each layer gets TERM when its parent dies, so a script killed with SIGKILL (no trap runs)
#    still takes its simulations down: timeout gets TERM, passes it to the command, and the
#    command gets TERM when timeout ends. The command runs in the background of the script,
#    which waits for it, so that the traps of 2 run at once on a signal.
#
# A parent which dies before its child has set the signal of its parent's death (before the
# prctl of setpriv) would leave the child without it: the relay of 1, the script and timeout of
# 3 therefore start through PROC_GUARD, which, with the signal set, ends the child if its parent
# is no longer the one which started it (for the relay: the parent its shell started with).

# `PROC_GUARD <pid> <cmd...>` (the script of `bash -c`, after `setpriv --pdeathsig`): <cmd...> in
# place of the shell if its parent is still <pid>, else the status of a TERM. A parent which
# dies after the check sends the signal of its death.
# shellcheck disable=SC2016 # expanded by the shell of the guard
PROC_GUARD='[[ $PPID == "$1" ]] || exit 143; shift; exec "$@"'

if [[ $(ps -o sid= -p $$ | tr -d ' ') != "$$" ]]; then
    # The relay of 1: this process, the PID the caller knows, in the caller's group (perhaps with
    # the rest of a pipeline, `script | tail`, which the cleanup is not to kill). Its pdeathsig
    # ends it with the caller (pdeathsig survives exec); the signals of the caller and of a
    # terminal (INT, TERM, HUP) end it by their default action; the script, its child in a
    # session of its own, gets TERM from its own pdeathsig when the relay ends. The script must
    # not stay in this process (`setsid` without a fork): a new session leaves the terminal's
    # foreground group, which Ctrl-C reaches, and a wrapper which waits for the script (`bash -c
    # "script; next"`) would wait out a Ctrl-C. The script runs by this bash, so it needs no exec
    # bit (`bash script.sh`).
    exec setpriv --pdeathsig TERM "$BASH" -c "$PROC_GUARD" proc-guard "$PPID" \
        setsid --fork --wait \
        setpriv --pdeathsig TERM "$BASH" -c "$PROC_GUARD" proc-guard "$$" "$BASH" "$0" "$@"
fi

PROC_PGID=$$
PROC_GRACE=${PROC_GRACE:-5}
PROC_CPUS=${PROC_CPUS:-0-9}

# The processes to take down: those of the group and every descendant of this shell (timeout
# makes a group of its own, mpiexec one per rank), but this shell, and the subshell this runs in
# (of `$(proc_victims)`) with its pipeline, which have ended when the list is used.
proc_victims() {
    ps -eo pid=,ppid=,pgid= | awk -v root="$$" -v self="$BASHPID" -v group="$PROC_PGID" '
        BEGIN { if (self == root) self = -1 }
        { parent[$1] = $2; pgid[$1] = $3 }
        END { for (p in parent) {
                  if (p == root || p == self) continue
                  q = p; n = 0
                  while (q in parent && q != root && q != self && q > 1 && n++ < 64) q = parent[q]
                  if (q != self && (q == root || pgid[p] == group)) print p } }'
}

proc_cleanup() {
    local status=$?
    # Without the script's errexit: a kill of a process which ended meanwhile (the list holds the
    # commands which listed it) fails, and must neither end the cleanup nor set the status.
    set +e
    trap - EXIT INT TERM HUP
    trap '' TERM
    local victims pid alive waited=0
    victims=$(proc_victims)
    if [[ -n $victims ]]; then
        # shellcheck disable=SC2086 # a list of pids
        kill -TERM $victims 2>/dev/null
        kill -TERM -- "-$PROC_PGID" 2>/dev/null
        while ((waited < PROC_GRACE * 10)); do
            alive=
            for pid in $victims; do kill -0 "$pid" 2>/dev/null && alive="$alive $pid"; done
            [[ -z $alive ]] && break
            sleep 0.1
            waited=$((waited + 1))
        done
        # shellcheck disable=SC2086 # a list of pids
        [[ -n $alive ]] && kill -KILL $alive 2>/dev/null
    fi
    exit "$status"
}
trap proc_cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

# run_sim <limit> <cmd...>: see above. Its status is that of the command (124 at the limit). The
# command runs in the background and the script waits for it: a shell runs its traps only
# between commands, and `wait` is the one a signal interrupts.
run_sim() {
    # This shell, the parent of the command (expanded here: the words of a background command
    # are expanded in its child).
    local limit=$1 parent=$BASHPID
    shift
    setpriv --pdeathsig TERM "$BASH" -c "$PROC_GUARD" proc-guard "$parent" \
        timeout --kill-after=10s "$limit" \
        setpriv --pdeathsig TERM taskset -c "$PROC_CPUS" nice -n 10 "$@" &
    wait "$!"
}
