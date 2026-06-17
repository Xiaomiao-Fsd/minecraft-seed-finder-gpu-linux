#!/usr/bin/env bash
# Compile and run the Witch Hut lowest-Y postprocessor against a filtered seed CSV.
#
# Default backend is CUDA, matching the Linux GPU-only pipeline style.
# Example:
#   scripts/find_lowest_witch_huts.sh \
#     --backend cuda \
#     --in runs/default/results_all_biomes.csv \
#     --out runs/default/witch_huts_negative_y.csv \
#     --whole-world \
#     --negative-y-only \
#     --top 50
#
# CPU fallback is explicit:
#   scripts/find_lowest_witch_huts.sh --backend cpu --in ... --out ...
#
# Set CUBIOMES_DIR if cubiomes is not at ../../tools/cubiomes and --backend cpu is used.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="${MC_WITCH_HUT_BACKEND:-cuda}"
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --backend requires cuda|cpu|auto" >&2
        exit 2
      fi
      BACKEND="$2"
      shift 2
      ;;
    --backend=*)
      BACKEND="${1#--backend=}"
      shift
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

find_cubiomes_dir() {
  if [[ -n "${CUBIOMES_DIR:-}" ]]; then
    printf '%s\n' "$CUBIOMES_DIR"
    return 0
  fi
  local candidates=(
    "$ROOT/../../tools/cubiomes"
    "$ROOT/../tools/cubiomes"
    "$ROOT/tools/cubiomes"
    "$HOME/.local/src/cubiomes"
  )
  local d
  for d in "${candidates[@]}"; do
    if [[ -r "$d/finders.h" && -r "$d/generator.h" ]]; then
      printf '%s\n' "$d"
      return 0
    fi
  done
  return 1
}

run_cuda() {
  local src="$ROOT/witch_hut_y_filter_cuda.cu"
  local bin="$ROOT/witch_hut_y_filter_cuda"
  if [[ ! -r "$src" ]]; then
    echo "ERROR: CUDA helper source missing: $src" >&2
    exit 1
  fi
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: nvcc not found; install CUDA Toolkit or rerun with --backend cpu." >&2
    exit 1
  fi
  if [[ ! -x "$bin" || "$src" -nt "$bin" || "$ROOT/biome_noise_cuda.h" -nt "$bin" || "$ROOT/minecraft_26_2_biome_params_cuda.h" -nt "$bin" ]]; then
    echo "[witch-hut] compiling CUDA helper: $bin" >&2
    nvcc -O3 -std=c++17 -o "$bin" "$src"
  fi
  exec "$bin" "${ARGS[@]}"
}

run_cpu() {
  local src="$ROOT/witch_hut_y_filter.c"
  local bin="$ROOT/witch_hut_y_filter"
  if [[ ! -r "$src" ]]; then
    echo "ERROR: CPU helper source missing: $src" >&2
    exit 1
  fi
  local cubiomes
  if ! cubiomes="$(find_cubiomes_dir)"; then
    cat >&2 <<'EOF'

Cubiomes was not found.
Install or clone cubiomes, then run with CUBIOMES_DIR, for example:

  CUBIOMES_DIR=/path/to/cubiomes scripts/find_lowest_witch_huts.sh \
    --backend cpu \
    --in runs/default/results_all_biomes.csv \
    --out runs/default/witch_huts_lowest.csv

EOF
    exit 1
  fi
  local lib="$cubiomes/libcubiomes.a"
  if [[ ! -r "$lib" ]]; then
    echo "[witch-hut] building cubiomes static library: $lib" >&2
    if ! make -C "$cubiomes" libcubiomes.a; then
      make -C "$cubiomes"
    fi
  fi
  if [[ ! -r "$lib" ]]; then
    echo "ERROR: missing cubiomes library after build: $lib" >&2
    exit 1
  fi
  if [[ ! -x "$bin" || "$src" -nt "$bin" || "$lib" -nt "$bin" ]]; then
    echo "[witch-hut] compiling CPU helper: $bin" >&2
    gcc -O3 -march=native -std=c11 -Wall -Wextra -fwrapv \
      -I"$cubiomes" \
      -o "$bin" "$src" "$lib" -lm
  fi
  exec "$bin" "${ARGS[@]}"
}

case "$BACKEND" in
  cuda)
    run_cuda
    ;;
  cpu)
    run_cpu
    ;;
  auto)
    if command -v nvcc >/dev/null 2>&1 && [[ -r "$ROOT/witch_hut_y_filter_cuda.cu" ]]; then
      run_cuda
    else
      run_cpu
    fi
    ;;
  *)
    echo "ERROR: unknown --backend '$BACKEND' (expected cuda|cpu|auto)" >&2
    exit 2
    ;;
esac
