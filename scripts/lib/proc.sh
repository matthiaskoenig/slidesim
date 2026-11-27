# shellcheck shell=bash
# Process hygiene for the scripts which run simulations: no process they start outlives them.
#
# Source it at the top of a script, before anything else runs, without arguments (it re-executes
# the script with the script's own arguments):
#
#   source "$(dirname "${BASH_SOURCE[0]}")/../lib/proc.sh"
#
# 1. The script runs in a process group (and session) of its own: unless it leads one, it
#    re-executes itself under `setsid`; in the same process if it does not lead the group it is
#    in, else in a child which gets TERM when the process the caller started ends.
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

if [[ $(ps -o sid= -p $$ | tr -d ' ') != "$$" ]]; then
    if [[ $(ps -o pgid= -p $$ | tr -d ' ') != "$$" ]]; then
        # Not a group leader, so setsid(2) succeeds in this process: exec keeps the PID.
        exec setsid "$0" "$@"
    fi
    # The leader of a group the caller made, perhaps with the rest of a pipeline (`script | tail`,
    # under job control), which the cleanup is not to kill: the script goes on in a child in a
    # session of its own, which gets TERM when this process ends (a signal to the PID the caller
    # knows, or Ctrl-C of a terminal, reaches the script through it).
    exec setsid -w setpriv --pdeathsig TERM "$0" "$@"
fi

PROC_PGID=$$
PROC_GRACE=${PROC_GRACE:-5}
PROC_CPUS=${PROC_CPUS:-0-9}

# The processes to take down: those of the group and every descendant of this shell (timeout
# makes a group of its own, mpiexec one per rank), but this shell and the subshell this runs in.
proc_victims() {
    { pgrep -g "$PROC_PGID"
      ps -eo pid=,ppid= | awk -v root="$$" '{ parent[$1] = $2 }
          END { for (p in parent) { q = p; n = 0
                    while (q in parent && q != root && q > 1 && n++ < 64) q = parent[q]
                    if (q == root) print p } }'
    } | sort -un | while read -r pid; do
        [[ $pid != "$$" && $pid != "$BASHPID" ]] && echo "$pid"
    done
    return 0
}

proc_cleanup() {
    local status=$?
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
    local limit=$1
    shift
    setpriv --pdeathsig TERM timeout --kill-after=10s "$limit" \
        setpriv --pdeathsig TERM taskset -c "$PROC_CPUS" nice -n 10 "$@" &
    wait "$!"
}
