#!/usr/bin/env bash
# Kill check of scripts/lib/proc.sh: no process a script started outlives it. Needs no slidesim,
# takes under a minute, and exits nonzero on any failure.
#
# A dummy script sources proc.sh and runs `run_sim 1m` of a command with a grandchild (two
# `sleep`s of a duration unique to the case). Per case the script is stopped, and within 15 s
# neither the dummy script (the relay and the script itself, found by the unique folder of the
# dummy), nor any of its `sleep`s may remain:
#   wrap-TERM, wrap-KILL   the signal to a wrapper `bash -c "dummy; echo done"`
#   wrap-INT               SIGINT to the process group of the wrapper (Ctrl-C of a terminal)
#   direct-TERM, direct-KILL  the signal to the dummy script itself
# and the exit status of the script: 0 for a normal run, 1 for a failing command, nonzero (124)
# for a time-limit hit. Processes are found by their unique marker; a foreign process is never
# signalled.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DIR=$(mktemp -d "${TMPDIR:-/tmp}/proc_kill.XXXXXX")
failures=0

# The processes whose command line holds a marker, other than this shell.
marked() { pgrep -f -- "$1" | grep -vx "$$" || true; }

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
    local cmd="sleep 600.$n & sleep 600.$n"
    case $case in
    wrap-*) set -m; bash -c "$dummy 1m bash -c '$cmd'; echo done" >"$log" 2>&1 & pid=$!; set +m ;;
    direct-*) "$dummy" 1m bash -c "$cmd" >"$log" 2>&1 & pid=$! ;;
    esac
    if ! wait_for 2 "sleep 600\\.$n"; then
        fail "$case: the sleeps did not start"
        return
    fi
    pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
    case $case in
    *-TERM) kill -TERM "$pid" ;;
    *-KILL) kill -KILL "$pid" ;;
    *-INT) kill -INT -- "-$pgid" ;;
    esac
    for i in $(seq 1 30); do
        left=$(( $(marked "sleep 600\\.$n" | wc -l) + $(marked "$dummy" | wc -l) ))
        ((left == 0)) && break
        sleep 0.5
    done
    { wait "$pid"; } 2>/dev/null
    if ((left == 0)); then
        echo "ok   $case: nothing left after $((i / 2)) s"
    else
        fail "$case: $left processes left: $(marked "sleep 600\\.$n" | tr '\n' ' ')$(marked "$dummy" | tr '\n' ' ')"
    fi
    # A wrapper which ends with its script goes on (prints done) in every case but a KILL of it.
    return 0
}

# status_case <name> <expected: zero|one|nonzero> <limit> <cmd...>
status_case() {
    local name=$1 want=$2 limit=$3 status
    shift 3
    "$dummy" "$limit" "$@" >"$DIR/$name.log" 2>&1
    status=$?
    case $want in
    zero) ((status == 0)) ;;
    one) ((status == 1)) ;;
    nonzero) ((status != 0)) ;;
    esac && echo "ok   $name: status $status" || fail "$name: status $status, wanted $want"
}

for c in wrap-TERM wrap-KILL wrap-INT direct-TERM direct-KILL; do kill_case "$c"; done
status_case normal zero 1m sleep 0.2
status_case failing one 1m false
status_case time-limit nonzero 2s sleep "600.$SUFFIX"
((${#SUFFIX} > 0 && $(marked "sleep 600\\.$SUFFIX" | wc -l) == 0)) || fail "time-limit: sleep left"

if ((failures > 0)); then
    echo "proc_kill: $failures failures"
    exit 1
fi
echo "proc_kill: all passed"
