#!/bin/bash
# HTCondor wrapper: run one training configuration (arguments are passed to hpt.train).
# The conda env (conda-pack tarball) and the dataset are copied to local scratch with xrdcp
# instead of read through the EOS fuse mount: ~100 jobs importing TF from EOS at once gives EIO.
# site settings (HPT_PROJECT, HPT_XRD_BASE, HPT_ENV_TARBALL): condor/site.env, copied next to this
# script by submit.sh (see condor/site.env.example)
HERE=$(cd "$(dirname "$0")" && pwd)
[ -f "$HERE/site.env" ] && source "$HERE/site.env"
PROJECT=${HPT_PROJECT:?set HPT_PROJECT in condor/site.env}
XRD=${HPT_XRD_BASE:?set HPT_XRD_BASE in condor/site.env}
ENV_TARBALL=${HPT_ENV_TARBALL:-env/hpt-env.tar.gz}
SCR=${_CONDOR_SCRATCH_DIR:-$(mktemp -d)}
NCPU=${_CONDOR_NUM_CPUS:-4}
export HPT_THREADS=$NCPU OMP_NUM_THREADS=$NCPU TF_NUM_INTRAOP_THREADS=$NCPU TF_CPP_MIN_LOG_LEVEL=2
export PYTHONDONTWRITEBYTECODE=1
echo "host $(hostname) cpus $NCPU start $(date)"

fetch() {  # fetch <remote> <local>, with retries
  for i in 1 2 3 4 5; do xrdcp -s -f "$1" "$2" && return 0; sleep $((20 + RANDOM % 60)); done
  return 1
}
sleep $((RANDOM % 30))
fetch "$XRD/$ENV_TARBALL" "$SCR/env.tgz" || { echo "env fetch failed" >&2; exit 1; }
mkdir -p "$SCR/env" && tar xzf "$SCR/env.tgz" -C "$SCR/env" && rm -f "$SCR/env.tgz"
source "$SCR/env/bin/activate" && conda-unpack
fetch "$XRD/data/jets_hlf.npz" "$SCR/jets_hlf.npz" || { echo "data fetch failed" >&2; exit 1; }
echo "env ready $(date)"

cd "$PROJECT" || exit 1
run_one() {
  for attempt in 1 2 3; do
    python -m hpt.train "$@" --data "$SCR/jets_hlf.npz" && return 0
    echo "attempt $attempt failed for: $*" >&2
    sleep $((30 + RANDOM % 60))
  done
  return 1
}
if [ "$1" = "--chunk" ]; then
  cp "$2" "$SCR/chunk.txt" || exit 1
  nfail=0
  while read -r line; do
    [ -z "$line" ] && continue
    # shellcheck disable=SC2086
    run_one $line || nfail=$((nfail + 1))
  done < "$SCR/chunk.txt"
  echo "end $(date), failed runs: $nfail"
  [ "$nfail" = "0" ] || exit 1
  exit 0
fi
run_one "$@" && { echo "end $(date)"; exit 0; }
exit 1
