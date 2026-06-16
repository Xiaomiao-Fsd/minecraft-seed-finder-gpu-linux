#!/usr/bin/env bash
# One-click launcher for the resumable multi-GPU Minecraft seed search runner.
set -euo pipefail

cd "$(dirname "$0")"

usage() {
  cat <<'EOF'
Usage:
  ./run_seed_cluster.sh                 # interactive: ask seed count, then start
  ./run_seed_cluster.sh 100000000       # start a new run scanning 100,000,000 seeds
  ./run_seed_cluster.sh 1b              # suffixes supported: k, m, g/b, t

Management:
  ./run_seed_cluster.sh status RUN_DIR
  ./run_seed_cluster.sh pause RUN_DIR
  ./run_seed_cluster.sh resume RUN_DIR
  ./run_seed_cluster.sh merge RUN_DIR

Environment overrides:
  MC_SEED_BACKEND=auto|cuda|cpu          default: cuda
  MC_SEED_GPUS=auto|0,1,2,3              default: auto
  MC_SEED_CHUNK_SIZE=10000000            default: 10000000
  MC_SEED_START=0                        default: 0
  MC_SEED_RUN_NAME=my_run                optional
  MC_SEED_CONFIG=config.example.json     default: config.example.json

Example for a 4x V100 node:
  MC_SEED_BACKEND=cuda MC_SEED_GPUS=0,1,2,3 ./run_seed_cluster.sh 1b
EOF
}

mode="${1:-}"
case "$mode" in
  -h|--help|help)
    usage
    exit 0
    ;;
  status|pause|resume|merge)
    if [[ $# -lt 2 ]]; then
      echo "Missing RUN_DIR" >&2
      usage >&2
      exit 2
    fi
    exec python3 seed_cluster_runner.py "$mode" --run-dir "$2"
    ;;
esac

count="${1:-}"
if [[ -z "$count" ]]; then
  read -r -p "How many seeds? Example: 100000000 or 1b: " count
fi
if [[ -z "$count" ]]; then
  echo "Seed count cannot be empty" >&2
  exit 2
fi

backend="${MC_SEED_BACKEND:-cuda}"
gpus="${MC_SEED_GPUS:-auto}"
chunk_size="${MC_SEED_CHUNK_SIZE:-10000000}"
start_seed="${MC_SEED_START:-0}"
config="${MC_SEED_CONFIG:-config.example.json}"

args=(
  start
  --count "$count"
  --start "$start_seed"
  --chunk-size "$chunk_size"
  --backend "$backend"
  --gpus "$gpus"
  --config "$config"
)

if [[ -n "${MC_SEED_RUN_NAME:-}" ]]; then
  args+=(--run-name "$MC_SEED_RUN_NAME")
fi

exec python3 seed_cluster_runner.py "${args[@]}"
