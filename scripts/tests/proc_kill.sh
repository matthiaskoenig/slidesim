#!/usr/bin/env bash
# Kill check of scripts/lib/proc.sh: no process a script started outlives it. Needs no slidesim,
# takes under a minute, and exits nonzero on any failure.
#
# A dummy script sources proc.sh and runs `run_sim 1m` of a command with a grandchild (two
# `sleep`s of a duration unique to the case). Per case the script is stopped once both sleeps
# run, and within 15 s no process of the case may remain (the wrapper, the relay and the script
# itself, the layers of run_sim and the sleeps: every one has the unique duration in its command
# line):
#   wrap-TERM, wrap-KILL   the signal to a wrapper `bash -c "dummy; echo done"`
#   wrap-INT               SIGINT to the process group of the wrapper (Ctrl-C of a terminal)
#   direct-TERM, direct-KILL  the signal to the dummy script itself
#   early-TERM             TERM to the wrapper while the dummy's shell has started but not yet
#                          read its script (a FIFO, written after the wrapper has ended), so
#                          before the relay could set the signal of its parent's death
#                          (PROC_GUARD of proc.sh)
# and the exit status of the script: 0 for a normal run, 1 for a failing command, nonzero (124)
# for a time-limit hit; 0 and 1 again for the script started as `bash <script>` without its exec
# bit. Processes are found by their unique marker; a foreign process is never signalled.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DIR=$(mktemp -d "${TMPDIR:-/tmp}/proc_kill.XXXXXX")
failures=0

# The processes whose command line holds a marker (an extended regular expression), other than
# this shell.
marked() { pgrep -f -- "$1" | grep -vx "$$" || true; }

# The processes `pids` (one per line) with their parents, groups, sessions and command lines.
listing() { [[ -n $1 ]] && ps -o pid=,ppid=,pgid=,sid=,args= -p "$(paste -sd, <<<"$1")"; }

cleanup() {
    local pid
    for pid in $(marked "$DIR") $(marked "sleep 600\\.$SUFFIX"); do kill -KILL "$pid" 2>/dev/null; done
    rm -rf "$DIR"
}
SUFFIX=$RANDOM$RANDOM
trap cleanup EXIT

dummy=$DIR/dummy.sh
cat >"$dummy" <<DUMMY
#!/usr/bin/env bash
# shellcheck source=scripts/lib/proc.sh
source "$ROOT/scripts/lib/proc.sh"
set -e
limit=\$1
shift
run_sim "\$limit" "\$@"
DUMMY
chmod +x "$dummy"
# The same script without the exec bit, which a caller starts as `bash <script>`.
plain=$DIR/plain.sh
cp "$dummy" "$plain"
chmod -x "$plain"

fail() { echo "FAIL $*"; failures=$((failures + 1)); }

# wait_for <count> <marker>: until that many processes hold the marker, at most 10 s.
wait_for() {
    local i
    for i in $(seq 1 100); do
        (($(marked "$2" | wc -l) >= $1)) && return 0
        sleep 0.1
    done
    return 1
}

kill_case() {
    local case=$1 n=$RANDOM$RANDOM pid pgid left i
    local log=$DIR/$case.log
    # The command of the run: a child with a grandchild, both sleeping for the unique duration.
    local cmd="sleep 600.$n & sleep 600.$n" fifo=$DIR/early.$n
    case $case in
    wrap-*)
        set -m
        bash -c "$dummy 1m bash -c '$cmd'; echo done" >"$log" 2>&1 &
        pid=$!
        set +m
        ;;
    early-*)
        mkfifo "$fifo"
        set -m
        bash -c "bash $fifo 1m bash -c '$cmd'; echo done" >"$log" 2>&1 &
        pid=$!
        set +m
        ;;
    direct-*) "$dummy" 1m bash -c "$cmd" >"$log" 2>&1 & pid=$! ;;
    esac
    case $case in
    # The dummy's shell, which waits for its script.
    early-*) wait_for 1 "^bash $fifo 1m " ;;
    # The sleeps themselves, not the processes which only name them.
    *) wait_for 2 "^sleep 600\\.$n\$" ;;
    esac || {
        fail "$case: the run did not start"
        return
    }
    pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
    case $case in
    early-*)
        kill -TERM "$pid"
        { wait "$pid"; } 2>/dev/null
        timeout 5 cp "$dummy" "$fifo"
        ;;
    *-TERM) kill -TERM "$pid" ;;
    *-KILL) kill -KILL "$pid" ;;
    *-INT) kill -INT -- "-$pgid" ;;
    esac
    # The shell reports a job which a KILL ended ("Killed") while it polls: not to the log.
    for i in $(seq 1 30); do
        left=$(marked "600\\.$n")
        [[ -z $left ]] && break
        sleep 0.5
    done 2>/dev/null
    { wait "$pid"; } 2>/dev/null
    if [[ -z $left ]]; then
        echo "ok   $case: nothing left after $((i / 2)) s"
    else
        fail "$case: $(wc -l <<<"$left") processes left (pid, ppid, pgid, sid, command):"
        listing "$left"
    fi
    return 0
}

# status_case <name> <expected: zero|one|nonzero> <limit> <cmd...>; the dummy runs as
# $STARTER (default: executed itself).
status_case() {
    local name=$1 want=$2 limit=$3 status
    shift 3
    # shellcheck disable=SC2086 # STARTER is a command line
    ${STARTER:-"$dummy"} "$limit" "$@" >"$DIR/$name.log" 2>&1
    status=$?
    if case $want in
        zero) ((status == 0)) ;;
        one) ((status == 1)) ;;
        nonzero) ((status != 0)) ;;
        esac
    then
        echo "ok   $name: status $status"
    else
        fail "$name: status $status, wanted $want"
    fi
}

for c in wrap-TERM wrap-KILL wrap-INT direct-TERM direct-KILL early-TERM; do kill_case "$c"; done
status_case normal zero 1m sleep 0.2
status_case failing one 1m false
status_case time-limit nonzero 2s sleep "600.$SUFFIX"
# Started as `bash <script>` without the exec bit: the re-execution under setsid runs it by bash.
STARTER="bash $plain" status_case bash-normal zero 1m sleep 0.2
STARTER="bash $plain" status_case bash-failing one 1m false
((${#SUFFIX} > 0 && $(marked "sleep 600\\.$SUFFIX" | wc -l) == 0)) || fail "time-limit: sleep left"
left=$(marked "$DIR")
[[ -z $left ]] || { fail "processes of the dummy left at the end:"; listing "$left"; }

if ((failures > 0)); then
    echo "proc_kill: $failures failures"
    exit 1
fi
echo "proc_kill: all passed"
