#!/bin/bash
# Usage: condor/submit.sh sweeps/<name>.args   (logs + submit inputs on AFS: ~/hpt_condor_logs/<name>)
set -e
SRC=$(dirname "$(readlink -f "$0")")
ARGS_FILE=$(readlink -f "$1")
NAME=$(basename "$ARGS_FILE" .args)
[ -f "$SRC/site.env" ] && source "$SRC/site.env"
LOGDIR=${HPT_LOGDIR:-$HOME/hpt_condor_logs}/$NAME
mkdir -p "$LOGDIR"
N=$(grep -c . "$ARGS_FILE" || true)
if [ "$N" = "0" ]; then echo "nothing to submit"; exit 0; fi
cp "$SRC/run.sh" "$LOGDIR/run.sh"; chmod +x "$LOGDIR/run.sh"
[ -f "$SRC/site.env" ] && cp "$SRC/site.env" "$LOGDIR/site.env"
cp "$ARGS_FILE" "$LOGDIR/jobs.args"
cp "$SRC/sweep.sub" "$LOGDIR/sweep.sub"
cd "$LOGDIR"
condor_submit -batch-name "hpt_$NAME" LOGDIR="$LOGDIR" sweep.sub
echo "submitted $N jobs from $ARGS_FILE, logs in $LOGDIR"
