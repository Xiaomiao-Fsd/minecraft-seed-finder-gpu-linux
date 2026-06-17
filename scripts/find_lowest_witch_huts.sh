#!/usr/bin/env bash
# Compile and run witch_hut_y_filter against a filtered seed CSV.
#
# Example:
#   scripts/find_lowest_witch_huts.sh \
#     --in runs/default/results_all_biomes.csv \
#     --out runs/default/witch_huts_lowest.csv \
#     --radius-blocks 60000 \
#     --top 50
#
# Set CUBIOMES_DIR if cubiomes is not at ../../tools/cubiomes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HELPER_SRC="$ROOT/witch_hut_y_filter.c"
HELPER_BIN="$ROOT/witch_hut_y_filter"

usage_extra() {
  cat >&2 <<'EOF'

Cubiomes was not found.
Install or clone cubiomes, then run with CUBIOMES_DIR, for example:

  CUBIOMES_DIR=/path/to/cubiomes scripts/find_lowest_witch_huts.sh \
    --in runs/default/results_all_biomes.csv \
    --out runs/default/witch_huts_lowest.csv

EOF
}

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

if [[ ! -r "$HELPER_SRC" ]]; then
  echo "ERROR: helper source missing: $HELPER_SRC" >&2
  exit 1
fi

if ! CUBIOMES="$(find_cubiomes_dir)"; then
  usage_extra
  exit 1
fi

LIB="$CUBIOMES/libcubiomes.a"
if [[ ! -r "$LIB" ]]; then
  echo "[witch-hut] building cubiomes static library: $LIB" >&2
  if ! make -C "$CUBIOMES" libcubiomes.a; then
    make -C "$CUBIOMES"
  fi
fi

if [[ ! -r "$LIB" ]]; then
  echo "ERROR: missing cubiomes library after build: $LIB" >&2
  exit 1
fi

if [[ ! -x "$HELPER_BIN" || "$HELPER_SRC" -nt "$HELPER_BIN" || "$LIB" -nt "$HELPER_BIN" ]]; then
  echo "[witch-hut] compiling $HELPER_BIN" >&2
  gcc -O3 -march=native -std=c11 -Wall -Wextra -fwrapv \
    -I"$CUBIOMES" \
    -o "$HELPER_BIN" "$HELPER_SRC" "$LIB" -lm
fi

exec "$HELPER_BIN" "$@"
