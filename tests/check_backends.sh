#!/usr/bin/env bash
# Every backend must leave the lattice in the bit-identical final state.
# Covers OpenMP thread counts and MPI rank counts, including an uneven split
# (L = 50 over 3 ranks gives strips of 17, 17 and 16 rows).
set -euo pipefail
BIN=${1:-build}
fail=0
for kernel in batched blocked baseline; do
for L in 64 50; do
  args="--L $L --T 2.3 --sweeps 300 --therm 0 --every 0 --seed 2024 --kernel $kernel"
  ref=$(OMP_NUM_THREADS=1 "$BIN/ising" $args | sed 's/.*checksum=//')
  echo "kernel=$kernel L=$L reference checksum $ref"
  for t in 2 4; do
    got=$(OMP_NUM_THREADS=$t "$BIN/ising" $args | sed 's/.*checksum=//')
    [[ "$got" == "$ref" ]] && echo "  openmp threads=$t  match" || { echo "  openmp threads=$t  MISMATCH $got"; fail=1; }
  done
  if [[ -x "$BIN/ising_mpi" ]]; then
    for r in 1 2 3; do
      got=$(OMP_NUM_THREADS=1 mpirun --oversubscribe --allow-run-as-root -np $r "$BIN/ising_mpi" $args | sed 's/.*checksum=//')
      [[ "$got" == "$ref" ]] && echo "  mpi ranks=$r        match" || { echo "  mpi ranks=$r        MISMATCH $got"; fail=1; }
    done
  fi
done
done
# batched and blocked kernels share one random stream: they must agree exactly
a=$(OMP_NUM_THREADS=1 "$BIN/ising" --L 64 --T 2.3 --sweeps 300 --therm 0 --every 0 --seed 2024 --kernel batched | sed 's/.*checksum=//')
b=$(OMP_NUM_THREADS=1 "$BIN/ising" --L 64 --T 2.3 --sweeps 300 --therm 0 --every 0 --seed 2024 --kernel blocked | sed 's/.*checksum=//')
[[ "$a" == "$b" ]] && echo "batched == blocked: match" || { echo "batched vs blocked MISMATCH"; fail=1; }
[[ $fail == 0 ]] && echo "backend consistency: passed" || { echo "backend consistency: FAILED"; exit 1; }
