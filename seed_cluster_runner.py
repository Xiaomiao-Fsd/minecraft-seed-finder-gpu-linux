#!/usr/bin/env python3
"""One-click resumable Minecraft seed search runner.

This orchestrates the slime-density prefilter over many GPUs by splitting the
seed range into restartable chunks. It is intentionally conservative: pause and
resume are chunk-boundary operations, so completed chunks are never rerun and an
interrupted running chunk is simply retried on resume.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import seed_finder

RUNS_DIR = ROOT / "runs"
STATE_FILE = "state.json"
PAUSE_FILE = "PAUSE"
DEFAULT_CHUNK_SIZE = 10_000_000
PROGRESS_RE = re.compile(r"scanned=(\d+)/(\d+)")


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def slug_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_state(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / STATE_FILE
    if not path.exists():
        raise FileNotFoundError(f"missing state file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(run_dir: Path, state: Dict[str, Any]) -> None:
    state["updated_at"] = now_iso()
    summarize_state_in_place(state)
    atomic_write_json(run_dir / STATE_FILE, state)


def parse_int(text: str, name: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{name} must be an integer: {text}")
    if value < 0:
        raise argparse.ArgumentTypeError(f"{name} must be >= 0")
    return value


def parse_count(text: str) -> int:
    multipliers = {"k": 1_000, "m": 1_000_000, "g": 1_000_000_000, "b": 1_000_000_000, "t": 1_000_000_000_000}
    raw = str(text).strip().lower().replace("_", "")
    if not raw:
        raise argparse.ArgumentTypeError("count is required")
    suffix = raw[-1]
    if suffix in multipliers:
        number = float(raw[:-1])
        value = int(number * multipliers[suffix])
    else:
        value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("count must be > 0")
    return value


def detect_gpus() -> List[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        ids = [item.strip() for item in visible.split(",") if item.strip()]
        if ids:
            return ids
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []
    try:
        proc = subprocess.run(
            [nvidia_smi, "--query-gpu=index", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return []
    if proc.returncode != 0:
        return []
    ids = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line:
            ids.append(line.split(",")[0].strip())
    return ids


def parse_gpus(value: str) -> List[str]:
    value = (value or "auto").strip()
    if value.lower() == "auto":
        return detect_gpus()
    if value.lower() in {"none", "cpu", ""}:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_backend(requested: str, gpus: List[str], *, uses_cuda: bool = True) -> str:
    requested = (requested or "auto").lower()
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"unsupported backend: {requested}")
    if not uses_cuda:
        if requested == "cuda":
            raise RuntimeError("configured prefilter does not have a CUDA backend")
        return "cpu"
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not gpus:
            raise RuntimeError("CUDA backend requested, but no GPU ids were detected/provided")
        # seed_finder.resolve_slime_backend checks nvcc and source availability.
        return seed_finder.resolve_slime_backend("cuda")
    # auto means CUDA if available; it must not silently fall back to CPU.
    if gpus and seed_finder.cuda_available():
        return "cuda"
    raise RuntimeError("auto backend could not find CUDA; pass --backend cpu explicitly for legacy CPU debugging")


def load_run_config(config_path: Path) -> Dict[str, Any]:
    return seed_finder.load_config(config_path)


def helper_command(config: Dict[str, Any], backend: str, start: int, count: int, out_csv: Path) -> List[str]:
    return seed_finder.prefilter_command(config, backend, start, count, out_csv)


def make_chunks(start_seed: int, count: int, chunk_size: int) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    total = int(math.ceil(count / float(chunk_size)))
    for index in range(total):
        chunk_start = start_seed + index * chunk_size
        chunk_count = min(chunk_size, start_seed + count - chunk_start)
        chunks.append({
            "index": index,
            "start": chunk_start,
            "count": chunk_count,
            "end_exclusive": chunk_start + chunk_count,
            "status": "pending",
            "attempts": 0,
            "output": f"chunks/chunk_{index:06d}.csv",
            "tmp_output": f"chunks/chunk_{index:06d}.csv.tmp",
            "log": f"logs/chunk_{index:06d}.log",
        })
    return chunks


def summarize_state_in_place(state: Dict[str, Any]) -> None:
    chunks = state.get("chunks", [])
    completed = [c for c in chunks if c.get("status") == "completed"]
    failed = [c for c in chunks if c.get("status") == "failed"]
    running = [c for c in chunks if c.get("status") == "running"]
    completed_count = sum(int(c.get("count", 0)) for c in completed)
    total_count = int(state.get("total_count", 0))
    completed_indices = {int(c["index"]) for c in completed}
    contiguous = 0
    while contiguous in completed_indices:
        contiguous += 1
    start_seed = int(state.get("start_seed", 0))
    chunk_size = int(state.get("chunk_size", DEFAULT_CHUNK_SIZE))
    highest_contiguous_seed = start_seed + min(total_count, contiguous * chunk_size) - 1 if contiguous > 0 else start_seed - 1
    current_estimate = highest_contiguous_seed
    for c in running:
        est = c.get("current_seed_estimate")
        if isinstance(est, int):
            current_estimate = max(current_estimate, est)
    state["summary"] = {
        "chunks_total": len(chunks),
        "chunks_completed": len(completed),
        "chunks_failed": len(failed),
        "chunks_running": len(running),
        "seeds_completed": completed_count,
        "total_count": total_count,
        "progress_percent": round((completed_count * 100.0 / total_count), 4) if total_count else 0.0,
        "highest_contiguous_completed_seed": highest_contiguous_seed,
        "current_seed_estimate": current_estimate,
    }


def create_run(args: argparse.Namespace) -> Path:
    config_path = Path(args.config).resolve()
    config = load_run_config(config_path)
    gpus = parse_gpus(args.gpus)
    uses_cuda = seed_finder.prefilter_supports_cuda(config)
    backend = resolve_backend(args.backend, gpus, uses_cuda=uses_cuda)
    if backend == "cpu":
        worker_count = args.workers or max(1, min(4, os.cpu_count() or 1))
    else:
        worker_count = len(gpus)
    if worker_count <= 0:
        raise RuntimeError("worker count resolved to 0")

    run_name = args.run_name or f"cluster_{slug_timestamp()}"
    run_dir = Path(args.run_dir).resolve() if args.run_dir else (RUNS_DIR / run_name).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"run directory already exists and is not empty: {run_dir}")
    (run_dir / "chunks").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "control").mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "config.json")

    state = {
        "version": 1,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "run_dir": str(run_dir),
        "status": "created",
        "prefilter_type": seed_finder.prefilter_type(config),
        "requested_backend": args.backend,
        "backend": backend,
        "gpus": gpus,
        "workers": worker_count,
        "start_seed": int(args.start),
        "total_count": int(args.count),
        "end_seed_exclusive": int(args.start) + int(args.count),
        "chunk_size": int(args.chunk_size),
        "max_retries": int(args.max_retries),
        "config_path": str(run_dir / "config.json"),
        "candidates_merged": "candidates_merged.csv",
        "chunks": make_chunks(int(args.start), int(args.count), int(args.chunk_size)),
    }
    save_state(run_dir, state)
    return run_dir


class ClusterRunner:
    def __init__(self, run_dir: Path, *, clear_pause: bool = False):
        self.run_dir = run_dir.resolve()
        self.state = load_state(self.run_dir)
        self.config = load_run_config(Path(self.state["config_path"]))
        self.lock = threading.Lock()
        self.stop_assigning = threading.Event()
        self.force_stop = threading.Event()
        self.active_procs: Dict[int, subprocess.Popen[str]] = {}
        self.pause_path = self.run_dir / "control" / PAUSE_FILE
        if clear_pause and self.pause_path.exists():
            self.pause_path.unlink()
        self._reset_stale_running_chunks()
        self._install_signal_handlers()

    def _install_signal_handlers(self) -> None:
        previous = {"count": 0}

        def handle_sigint(signum: int, frame: Any) -> None:  # noqa: ARG001
            previous["count"] += 1
            if previous["count"] == 1:
                print("\n[pause] Ctrl+C received. Will pause after current chunks finish. Press Ctrl+C again to terminate running chunks.", file=sys.stderr)
                self.request_pause()
            else:
                print("\n[stop] Second Ctrl+C received. Terminating running chunks; they will be retried on resume.", file=sys.stderr)
                self.force_stop.set()
                self.request_pause()
                with self.lock:
                    for proc in list(self.active_procs.values()):
                        if proc.poll() is None:
                            proc.terminate()

        signal.signal(signal.SIGINT, handle_sigint)
        signal.signal(signal.SIGTERM, handle_sigint)

    def _reset_stale_running_chunks(self) -> None:
        with self.lock:
            for chunk in self.state.get("chunks", []):
                if chunk.get("status") == "running":
                    chunk["status"] = "pending"
                    chunk.pop("worker", None)
                    chunk.pop("gpu", None)
                    chunk.pop("pid", None)
                    chunk.pop("current_seed_estimate", None)
            save_state(self.run_dir, self.state)

    def request_pause(self) -> None:
        self.pause_path.parent.mkdir(parents=True, exist_ok=True)
        self.pause_path.write_text(f"pause requested at {now_iso()}\n", encoding="utf-8")
        self.stop_assigning.set()

    def paused_requested(self) -> bool:
        return self.pause_path.exists() or self.stop_assigning.is_set()

    def next_chunk(self, worker_id: int, gpu_id: Optional[str]) -> Optional[Dict[str, Any]]:
        with self.lock:
            if self.paused_requested():
                return None
            max_retries = int(self.state.get("max_retries", 1))
            for chunk in self.state.get("chunks", []):
                status = chunk.get("status")
                attempts = int(chunk.get("attempts", 0))
                if status == "pending" or (status == "failed" and attempts < max_retries):
                    chunk["status"] = "running"
                    chunk["worker"] = worker_id
                    chunk["gpu"] = gpu_id
                    chunk["attempts"] = attempts + 1
                    chunk["started_at"] = now_iso()
                    chunk["current_seed_estimate"] = int(chunk["start"])
                    save_state(self.run_dir, self.state)
                    return dict(chunk)
        return None

    def mark_chunk_completed(self, chunk_index: int, elapsed: float, candidates: Optional[int]) -> None:
        with self.lock:
            chunk = self.state["chunks"][chunk_index]
            chunk["status"] = "completed"
            chunk["completed_at"] = now_iso()
            chunk["elapsed_seconds"] = elapsed
            if elapsed > 0:
                chunk["seeds_per_second"] = int(chunk["count"]) / elapsed
            if candidates is not None:
                chunk["candidates"] = candidates
            chunk["current_seed_estimate"] = int(chunk["end_exclusive"]) - 1
            for key in ["worker", "gpu", "pid"]:
                chunk.pop(key, None)
            save_state(self.run_dir, self.state)

    def mark_chunk_failed(self, chunk_index: int, returncode: int, error: str) -> None:
        with self.lock:
            chunk = self.state["chunks"][chunk_index]
            chunk["status"] = "failed"
            chunk["failed_at"] = now_iso()
            chunk["returncode"] = returncode
            chunk["error"] = error[-2000:]
            for key in ["worker", "gpu", "pid"]:
                chunk.pop(key, None)
            save_state(self.run_dir, self.state)
            self.stop_assigning.set()

    def update_chunk_progress(self, chunk_index: int, scanned: int) -> None:
        with self.lock:
            chunk = self.state["chunks"][chunk_index]
            if chunk.get("status") == "running":
                current = int(chunk["start"]) + min(int(scanned), int(chunk["count"])) - 1
                chunk["current_seed_estimate"] = max(int(chunk.get("current_seed_estimate", chunk["start"])), current)
                save_state(self.run_dir, self.state)

    def count_candidates(self, csv_path: Path) -> Optional[int]:
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                return max(0, sum(1 for _ in handle) - 1)
        except OSError:
            return None

    def run_chunk(self, worker_id: int, gpu_id: Optional[str], chunk: Dict[str, Any]) -> None:
        backend = self.state["backend"]
        out_final = self.run_dir / chunk["output"]
        out_tmp = self.run_dir / chunk["tmp_output"]
        log_path = self.run_dir / chunk["log"]
        out_tmp.parent.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if out_tmp.exists():
            out_tmp.unlink()
        cmd = helper_command(self.config, backend, int(chunk["start"]), int(chunk["count"]), out_tmp)
        env = os.environ.copy()
        if backend == "cuda" and gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        t0 = time.monotonic()
        with log_path.open("a", encoding="utf-8", errors="replace") as log:
            log.write(f"\n=== chunk {chunk['index']} worker={worker_id} gpu={gpu_id} start={chunk['start']} count={chunk['count']} at {now_iso()} ===\n")
            log.write("command: " + " ".join(cmd) + "\n")
            if backend == "cuda" and gpu_id is not None:
                log.write(f"CUDA_VISIBLE_DEVICES={gpu_id}\n")
            log.flush()
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                cwd=str(ROOT),
            )
            with self.lock:
                self.active_procs[worker_id] = proc
                self.state["chunks"][int(chunk["index"])] ["pid"] = proc.pid
                save_state(self.run_dir, self.state)
            last_lines: List[str] = []
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()
                last_lines.append(line.rstrip("\n"))
                if len(last_lines) > 20:
                    last_lines.pop(0)
                match = PROGRESS_RE.search(line)
                if match:
                    self.update_chunk_progress(int(chunk["index"]), int(match.group(1)))
            rc = proc.wait()
            with self.lock:
                self.active_procs.pop(worker_id, None)
        elapsed = time.monotonic() - t0
        if rc == 0 and out_tmp.exists():
            out_tmp.replace(out_final)
            candidates = self.count_candidates(out_final)
            self.mark_chunk_completed(int(chunk["index"]), elapsed, candidates)
        else:
            error = "\n".join(last_lines) if 'last_lines' in locals() else "process failed"
            if self.force_stop.is_set():
                # Interrupted chunks should be retried, not counted as hard failures.
                with self.lock:
                    st_chunk = self.state["chunks"][int(chunk["index"])]
                    st_chunk["status"] = "pending"
                    st_chunk["interrupted_at"] = now_iso()
                    for key in ["worker", "gpu", "pid"]:
                        st_chunk.pop(key, None)
                    save_state(self.run_dir, self.state)
            else:
                self.mark_chunk_failed(int(chunk["index"]), rc, error)

    def worker_loop(self, worker_id: int, gpu_id: Optional[str]) -> None:
        while not self.force_stop.is_set():
            chunk = self.next_chunk(worker_id, gpu_id)
            if chunk is None:
                return
            self.run_chunk(worker_id, gpu_id, chunk)

    def run(self) -> None:
        with self.lock:
            if self.state.get("status") == "completed":
                print(f"[done] already completed: {self.run_dir}")
                return
            self.state["status"] = "running"
            self.state["started_or_resumed_at"] = now_iso()
            save_state(self.run_dir, self.state)
        backend = self.state["backend"]
        gpus = list(self.state.get("gpus", []))
        workers = int(self.state.get("workers", len(gpus) or 1))
        print(f"[run] dir={self.run_dir}")
        print(f"[run] backend={backend} workers={workers} gpus={gpus or 'cpu'} chunk_size={self.state.get('chunk_size')}")
        print(f"[hint] pause from another shell: python3 seed_cluster_runner.py pause --run-dir {self.run_dir}")

        threads: List[threading.Thread] = []
        for worker_id in range(workers):
            gpu_id = gpus[worker_id % len(gpus)] if backend == "cuda" and gpus else None
            thread = threading.Thread(target=self.worker_loop, args=(worker_id, gpu_id), daemon=True)
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()

        merged_path: Optional[Path] = None
        run_postfilters = False
        with self.lock:
            chunks = self.state.get("chunks", [])
            if all(c.get("status") == "completed" for c in chunks):
                self.state["status"] = "completed"
                self.state["completed_at"] = now_iso()
                save_state(self.run_dir, self.state)
                merged_path = merge_candidates(self.run_dir)
                run_postfilters = True
                print(f"[done] completed. merged={merged_path}")
            elif self.paused_requested():
                self.state["status"] = "paused"
                save_state(self.run_dir, self.state)
                print(f"[paused] {self.run_dir}")
            else:
                self.state["status"] = "failed"
                save_state(self.run_dir, self.state)
                print(f"[failed] {self.run_dir}", file=sys.stderr)
        if run_postfilters and merged_path is not None:
            try:
                outputs = seed_finder.run_configured_postfilters(self.config, merged_path, run_dir=self.run_dir, backend=str(self.state.get("requested_backend", "auto")))
            except (RuntimeError, OSError, ValueError, subprocess.CalledProcessError) as exc:
                with self.lock:
                    self.state["status"] = "postfilter_failed"
                    self.state["postfilter_error"] = str(exc)
                    save_state(self.run_dir, self.state)
                print(f"[postfilter_failed] {exc}", file=sys.stderr)
                return
            if outputs:
                with self.lock:
                    self.state["postfilters"] = outputs
                    self.state["status"] = "completed"
                    save_state(self.run_dir, self.state)
                print(f"[done] postfilters={outputs}")


def merge_candidates(run_dir: Path) -> Path:
    state = load_state(run_dir)
    config = load_run_config(Path(state["config_path"]))
    out_path = run_dir / state.get("candidates_merged", "candidates_merged.csv")
    completed = [c for c in state.get("chunks", []) if c.get("status") == "completed"]
    completed.sort(key=lambda c: int(c["index"]))
    header_written = False
    with out_path.open("w", encoding="utf-8", newline="") as dst:
        writer = None
        for chunk in completed:
            path = run_dir / chunk["output"]
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8", newline="") as src:
                reader = csv.DictReader(src)
                if writer is None:
                    writer = csv.DictWriter(dst, fieldnames=reader.fieldnames or [])
                    writer.writeheader()
                    header_written = True
                for row in reader:
                    writer.writerow(row)
    if not header_written:
        out_path.write_text(seed_finder.prefilter_empty_header(config), encoding="utf-8")
    return out_path


def print_status(run_dir: Path, as_json: bool = False) -> None:
    state = load_state(run_dir)
    summarize_state_in_place(state)
    if as_json:
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return
    summary = state.get("summary", {})
    print(f"run_dir: {run_dir.resolve()}")
    print(f"status: {state.get('status')} prefilter={state.get('prefilter_type')} backend={state.get('backend')} workers={state.get('workers')} gpus={state.get('gpus') or 'cpu'}")
    print(f"range: start={state.get('start_seed')} count={state.get('total_count')} end_exclusive={state.get('end_seed_exclusive')}")
    print(f"progress: {summary.get('seeds_completed')}/{summary.get('total_count')} seeds ({summary.get('progress_percent')}%)")
    print(f"chunks: completed={summary.get('chunks_completed')}/{summary.get('chunks_total')} running={summary.get('chunks_running')} failed={summary.get('chunks_failed')}")
    print(f"highest_contiguous_completed_seed: {summary.get('highest_contiguous_completed_seed')}")
    print(f"current_seed_estimate: {summary.get('current_seed_estimate')}")
    if (run_dir / "control" / PAUSE_FILE).exists():
        print("pause_requested: yes")
    if state.get("postfilters"):
        print(f"postfilters: {state.get('postfilters')}")
    if state.get("postfilter_error"):
        print(f"postfilter_error: {state.get('postfilter_error')}")
    running = [c for c in state.get("chunks", []) if c.get("status") == "running"]
    for c in running[:8]:
        print(f"  worker={c.get('worker')} gpu={c.get('gpu')} chunk={c.get('index')} start={c.get('start')} current≈{c.get('current_seed_estimate')} end={int(c.get('end_exclusive', 0))-1}")


def cmd_start(args: argparse.Namespace) -> int:
    run_dir = create_run(args)
    runner = ClusterRunner(run_dir)
    runner.run()
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    runner = ClusterRunner(run_dir, clear_pause=True)
    runner.run()
    return 0


def cmd_pause(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    pause_path = run_dir / "control" / PAUSE_FILE
    pause_path.parent.mkdir(parents=True, exist_ok=True)
    pause_path.write_text(f"pause requested at {now_iso()}\n", encoding="utf-8")
    print(f"[pause] requested. Running chunks will finish before the runner exits: {run_dir}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    print_status(Path(args.run_dir), as_json=args.json)
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    out = merge_candidates(Path(args.run_dir))
    print(out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="One-click resumable multi-GPU Minecraft seed search runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_start = sub.add_parser("start", help="create a new run and start searching")
    p_start.add_argument("--count", type=parse_count, required=True, help="number of seeds to scan, supports 100m/1b suffixes")
    p_start.add_argument("--start", type=int, default=0, help="first seed, default 0")
    p_start.add_argument("--chunk-size", type=parse_count, default=DEFAULT_CHUNK_SIZE, help="checkpoint chunk size, default 10m")
    p_start.add_argument("--config", type=Path, default=seed_finder.DEFAULT_CONFIG)
    p_start.add_argument("--backend", choices=["auto", "cuda", "cpu"], default="cuda")
    p_start.add_argument("--gpus", default="auto", help="auto or comma-separated GPU ids, e.g. 0,1,2,3")
    p_start.add_argument("--workers", type=int, help="worker count for CPU/debug runs; CUDA uses one worker per GPU")
    p_start.add_argument("--run-name", help="run name under runs/")
    p_start.add_argument("--run-dir", help="explicit run directory")
    p_start.add_argument("--max-retries", type=int, default=2)
    p_start.set_defaults(func=cmd_start)

    p_resume = sub.add_parser("resume", help="resume an existing run directory")
    p_resume.add_argument("--run-dir", required=True)
    p_resume.set_defaults(func=cmd_resume)

    p_pause = sub.add_parser("pause", help="request pause for a running run directory")
    p_pause.add_argument("--run-dir", required=True)
    p_pause.set_defaults(func=cmd_pause)

    p_status = sub.add_parser("status", help="print run status")
    p_status.add_argument("--run-dir", required=True)
    p_status.add_argument("--json", action="store_true")
    p_status.set_defaults(func=cmd_status)

    p_merge = sub.add_parser("merge", help="merge completed chunk CSVs")
    p_merge.add_argument("--run-dir", required=True)
    p_merge.set_defaults(func=cmd_merge)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (RuntimeError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
