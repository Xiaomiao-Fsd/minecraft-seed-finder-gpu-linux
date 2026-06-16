#!/usr/bin/env python3
"""Minecraft Java seed search orchestration framework.

Default mode is GPU-only: CUDA quad ocean-monument structure candidates followed
by CUDA exact 17x17 slime-window scoring. Legacy CPU/cubiomes helpers are kept
only for explicit manual verification runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.example.json"


def helper_bin(name: str) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return ROOT / f"{name}{suffix}"


C_HELPER_SRC = ROOT / "slime_prefilter.c"
C_HELPER_BIN = helper_bin("slime_prefilter")
CUDA_HELPER_SRC = ROOT / "slime_prefilter_cuda.cu"
CUDA_HELPER_BIN = helper_bin("slime_prefilter_cuda")
MONUMENT_HELPER_SRC = ROOT / "ocean_monument_filter.c"
MONUMENT_HELPER_BIN = helper_bin("ocean_monument_filter")
MONUMENT_CUDA_HELPER_SRC = ROOT / "ocean_monument_quad_cuda.cu"
MONUMENT_CUDA_HELPER_BIN = helper_bin("ocean_monument_quad_cuda")
SLIME_WINDOW_HELPER_SRC = ROOT / "slime_window_filter.c"
SLIME_WINDOW_HELPER_BIN = helper_bin("slime_window_filter")
SLIME_WINDOW_CUDA_HELPER_SRC = ROOT / "slime_window_filter_cuda.cu"
SLIME_WINDOW_CUDA_HELPER_BIN = helper_bin("slime_window_filter_cuda")
ALL_BIOMES_HELPER_SRC = ROOT / "all_biomes_filter.c"
ALL_BIOMES_HELPER_BIN = helper_bin("all_biomes_filter")
ALL_BIOMES_CUDA_HELPER_SRC = ROOT / "all_biomes_cuda.cu"
ALL_BIOMES_CUDA_HELPER_BIN = helper_bin("all_biomes_cuda")


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def output_dir(config: Dict[str, Any]) -> Path:
    out = config.get("output", {})
    directory = Path(out.get("directory", "runs/default"))
    if not directory.is_absolute():
        directory = ROOT / directory
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def output_file(config: Dict[str, Any], key: str, default: str, run_dir: Optional[Path] = None) -> Path:
    out = config.get("output", {})
    path = Path(out.get(key, default))
    if path.is_absolute():
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    base = run_dir if run_dir is not None else output_dir(config)
    base.mkdir(parents=True, exist_ok=True)
    return base / path


def stage_config(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    if name == str(config.get("prefilter", {}).get("type", "")):
        return config.get("prefilter", {})
    postfilters = config.get("postfilters", {})
    if name in postfilters:
        return postfilters.get(name, {})
    checks = config.get("checks", {})
    return checks.get(name, {})


def stage_enabled(config: Dict[str, Any], name: str) -> bool:
    cfg = stage_config(config, name)
    return bool(cfg.get("enabled", True))


def minecraft_version_arg(config: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    return str(cfg.get("mc") or cfg.get("version") or config.get("target_version", "26.2"))


def prefilter_type(config: Dict[str, Any]) -> str:
    return str(config.get("prefilter", {}).get("type", "slime_density"))


def prefilter_supports_cuda(config: Dict[str, Any]) -> bool:
    kind = prefilter_type(config)
    return kind in {"slime_density", "quad_ocean_monument"}


def resolve_cubiomes_dir(config: Dict[str, Any]) -> Path:
    tools = config.get("tools", {})
    configured = tools.get("cubiomes_dir") or os.environ.get("CUBIOMES_DIR") or (ROOT / "tools" / "cubiomes")
    path = Path(configured)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    required = [
        path / "finders.h",
        path / "generator.h",
        path / "libcubiomes.a",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(
            "cubiomes is required for monument/biome filters. Set CUBIOMES_DIR or tools.cubiomes_dir; "
            f"missing: {', '.join(missing)}"
        )
    return path


def compile_cubiomes_helper(src: Path, out_bin: Path, config: Dict[str, Any], *, force: bool = False, extra_libs: Optional[List[str]] = None) -> Path:
    if not src.exists():
        raise FileNotFoundError(src)
    cubiomes = resolve_cubiomes_dir(config)
    libcubiomes = cubiomes / "libcubiomes.a"
    newest_input = max(src.stat().st_mtime, libcubiomes.stat().st_mtime)
    if out_bin.exists() and not force and out_bin.stat().st_mtime >= newest_input:
        return out_bin
    cmd = [
        "gcc",
        "-O3",
        "-march=native",
        "-std=c11",
        "-Wall",
        "-Wextra",
        "-fwrapv",
        "-I",
        str(cubiomes),
        "-o",
        str(out_bin),
        str(src),
        str(libcubiomes),
    ]
    cmd.extend(extra_libs or [])
    print("[build]", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)
    return out_bin


def compile_c_helper(force: bool = False) -> Path:
    if not C_HELPER_SRC.exists():
        raise FileNotFoundError(C_HELPER_SRC)
    if C_HELPER_BIN.exists() and not force and C_HELPER_BIN.stat().st_mtime >= C_HELPER_SRC.stat().st_mtime:
        return C_HELPER_BIN
    cmd = [
        "gcc",
        "-O3",
        "-march=native",
        "-std=c11",
        "-Wall",
        "-Wextra",
        "-o",
        str(C_HELPER_BIN),
        str(C_HELPER_SRC),
    ]
    print("[build]", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)
    return C_HELPER_BIN


def run_nvcc(args: List[str]) -> None:
    cmd = ["nvcc", *args]
    print("[build]", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)


def nvcc_compat_flags() -> List[str]:
    return []


def cuda_available() -> bool:
    return CUDA_HELPER_SRC.exists() and shutil.which("nvcc") is not None


def compile_cuda_helper(force: bool = False) -> Path:
    if not CUDA_HELPER_SRC.exists():
        raise FileNotFoundError(CUDA_HELPER_SRC)
    if shutil.which("nvcc") is None:
        raise RuntimeError("nvcc not found; install CUDA toolkit or use --backend cpu")
    if CUDA_HELPER_BIN.exists() and not force and CUDA_HELPER_BIN.stat().st_mtime >= CUDA_HELPER_SRC.stat().st_mtime:
        return CUDA_HELPER_BIN
    args = [
        *nvcc_compat_flags(),
        "-O3",
        "-std=c++17",
        "-o",
        str(CUDA_HELPER_BIN),
        str(CUDA_HELPER_SRC),
    ]
    run_nvcc(args)
    return CUDA_HELPER_BIN


def compile_ocean_monument_helper(config: Dict[str, Any], force: bool = False) -> Path:
    return compile_cubiomes_helper(MONUMENT_HELPER_SRC, MONUMENT_HELPER_BIN, config, force=force, extra_libs=["-lm"])


def compile_ocean_monument_cuda_helper(force: bool = False) -> Path:
    if not MONUMENT_CUDA_HELPER_SRC.exists():
        raise FileNotFoundError(MONUMENT_CUDA_HELPER_SRC)
    if shutil.which("nvcc") is None:
        raise RuntimeError("nvcc not found; install CUDA toolkit for quad_ocean_monument --backend cuda")
    inputs = [
        MONUMENT_CUDA_HELPER_SRC,
        ROOT / "biome_noise_cuda.h",
        ROOT / "minecraft_26_2_biome_params_cuda.h",
    ]
    newest_input = max(p.stat().st_mtime for p in inputs if p.exists())
    if MONUMENT_CUDA_HELPER_BIN.exists() and not force and MONUMENT_CUDA_HELPER_BIN.stat().st_mtime >= newest_input:
        return MONUMENT_CUDA_HELPER_BIN
    args = [
        *nvcc_compat_flags(),
        "-O3",
        "-std=c++17",
        "-o",
        str(MONUMENT_CUDA_HELPER_BIN),
        str(MONUMENT_CUDA_HELPER_SRC),
    ]
    run_nvcc(args)
    return MONUMENT_CUDA_HELPER_BIN


def compile_slime_window_helper(force: bool = False) -> Path:
    if not SLIME_WINDOW_HELPER_SRC.exists():
        raise FileNotFoundError(SLIME_WINDOW_HELPER_SRC)
    if SLIME_WINDOW_HELPER_BIN.exists() and not force and SLIME_WINDOW_HELPER_BIN.stat().st_mtime >= SLIME_WINDOW_HELPER_SRC.stat().st_mtime:
        return SLIME_WINDOW_HELPER_BIN
    cmd = [
        "gcc",
        "-O3",
        "-march=native",
        "-std=c11",
        "-Wall",
        "-Wextra",
        "-o",
        str(SLIME_WINDOW_HELPER_BIN),
        str(SLIME_WINDOW_HELPER_SRC),
    ]
    print("[build]", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)
    return SLIME_WINDOW_HELPER_BIN


def compile_slime_window_cuda_helper(force: bool = False) -> Path:
    if not SLIME_WINDOW_CUDA_HELPER_SRC.exists():
        raise FileNotFoundError(SLIME_WINDOW_CUDA_HELPER_SRC)
    if shutil.which("nvcc") is None:
        raise RuntimeError("nvcc not found; install CUDA toolkit for slime_window_density --backend cuda")
    if SLIME_WINDOW_CUDA_HELPER_BIN.exists() and not force and SLIME_WINDOW_CUDA_HELPER_BIN.stat().st_mtime >= SLIME_WINDOW_CUDA_HELPER_SRC.stat().st_mtime:
        return SLIME_WINDOW_CUDA_HELPER_BIN
    args = [
        *nvcc_compat_flags(),
        "-O3",
        "-std=c++17",
        "-o",
        str(SLIME_WINDOW_CUDA_HELPER_BIN),
        str(SLIME_WINDOW_CUDA_HELPER_SRC),
    ]
    run_nvcc(args)
    return SLIME_WINDOW_CUDA_HELPER_BIN


def compile_all_biomes_helper(config: Dict[str, Any], force: bool = False) -> Path:
    return compile_cubiomes_helper(ALL_BIOMES_HELPER_SRC, ALL_BIOMES_HELPER_BIN, config, force=force)


def compile_all_biomes_cuda_helper(force: bool = False) -> Path:
    if not ALL_BIOMES_CUDA_HELPER_SRC.exists():
        raise FileNotFoundError(ALL_BIOMES_CUDA_HELPER_SRC)
    if shutil.which("nvcc") is None:
        raise RuntimeError("nvcc not found; install CUDA toolkit for all_biomes_within_origin_radius --backend cuda")
    inputs = [
        ALL_BIOMES_CUDA_HELPER_SRC,
        ROOT / "biome_noise_cuda.h",
        ROOT / "minecraft_26_2_biome_params_cuda.h",
    ]
    newest_input = max(p.stat().st_mtime for p in inputs if p.exists())
    if ALL_BIOMES_CUDA_HELPER_BIN.exists() and not force and ALL_BIOMES_CUDA_HELPER_BIN.stat().st_mtime >= newest_input:
        return ALL_BIOMES_CUDA_HELPER_BIN
    args = [
        *nvcc_compat_flags(),
        "-O3",
        "-std=c++17",
        "-o",
        str(ALL_BIOMES_CUDA_HELPER_BIN),
        str(ALL_BIOMES_CUDA_HELPER_SRC),
    ]
    run_nvcc(args)
    return ALL_BIOMES_CUDA_HELPER_BIN


def resolve_slime_backend(requested: str) -> str:
    requested = (requested or "auto").lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"unsupported slime backend: {requested}")
    if requested == "auto":
        if cuda_available():
            return "cuda"
        raise RuntimeError("auto backend could not find CUDA; pass --backend cpu explicitly for legacy CPU debugging")
    if requested == "cuda" and not cuda_available():
        raise RuntimeError("CUDA backend requested, but nvcc/CUDA helper is not available")
    return requested


def slime_prefilter_command(
    config: Dict[str, Any],
    backend: str,
    start: int,
    count: int,
    out_csv: Path,
    *,
    force_build: bool = False,
) -> List[str]:
    pf = config.get("prefilter", {})
    circle_radius_chunks = max(1, int(round(float(pf.get("circle_radius_blocks", 128)) / 16.0)))
    search_radius_chunks = max(0, int(round(float(pf.get("search_radius_blocks", 10000)) / 16.0)))
    threshold = int(pf.get("threshold_chunks", 28))
    center_samples = int(pf.get("center_samples", 64))
    max_candidates = int(pf.get("max_candidates", 0))
    helper = compile_cuda_helper(force=force_build) if backend == "cuda" else compile_c_helper(force=force_build)
    cmd = [
        str(helper),
        "--start", str(start),
        "--count", str(count),
        "--threshold", str(threshold),
        "--circle-radius-chunks", str(circle_radius_chunks),
        "--search-radius-chunks", str(search_radius_chunks),
        "--center-samples", str(center_samples),
        "--max-candidates", str(max_candidates),
        "--out", str(out_csv),
    ]
    if backend == "cuda":
        cuda_cfg = pf.get("cuda", {})
        cmd += [
            "--batch-size", str(int(cuda_cfg.get("batch_size", 1048576))),
            "--threads", str(int(cuda_cfg.get("threads", 256))),
        ]
    return cmd


def quad_monument_prefilter_command(
    config: Dict[str, Any],
    backend: str,
    start: int,
    count: int,
    out_csv: Path,
    *,
    force_build: bool = False,
) -> List[str]:
    pf = config.get("prefilter", {})
    resolved = (backend or "auto").lower()
    if resolved == "cuda":
        helper = compile_ocean_monument_cuda_helper(force=force_build)
    else:
        helper = compile_ocean_monument_helper(config, force=force_build)
    cmd = [
        str(helper),
        "--start", str(start),
        "--count", str(count),
        "--out", str(out_csv),
        "--radius-blocks", str(int(pf.get("search_radius_blocks", pf.get("radius_blocks", 60000)))),
        "--cluster-window-chunks", str(int(pf.get("cluster_window_chunks", pf.get("window_chunks", 17)))),
        "--required-count", str(int(pf.get("required_monuments", 4))),
        "--footprint-blocks", str(int(pf.get("monument_footprint_blocks", 58))),
        "--center-offset-blocks", str(int(pf.get("monument_center_offset_blocks", 8))),
        "--mc", minecraft_version_arg(config, pf),
    ]
    if bool(pf.get("window_must_fit_inside_radius", True)):
        cmd.append("--window-inside-radius")
    if resolved == "cuda" and not bool(pf.get("check_biome_viability", True)):
        cmd.append("--skip-biome-viability")
    max_candidates = int(pf.get("max_candidates", 0))
    if max_candidates > 0:
        cmd += ["--max-candidates", str(max_candidates)]
    if resolved == "cuda":
        cuda_cfg = pf.get("cuda", {})
        cmd += [
            "--batch-size", str(int(cuda_cfg.get("batch_size", 1048576))),
            "--threads", str(int(cuda_cfg.get("threads", 256))),
        ]
    return cmd


def prefilter_command(
    config: Dict[str, Any],
    backend: str,
    start: int,
    count: int,
    out_csv: Path,
    *,
    force_build: bool = False,
) -> List[str]:
    kind = prefilter_type(config)
    if kind == "slime_density":
        resolved = resolve_slime_backend(backend or str(config.get("prefilter", {}).get("backend", "auto")))
        return slime_prefilter_command(config, resolved, start, count, out_csv, force_build=force_build)
    if kind == "quad_ocean_monument":
        resolved = (backend or "auto").lower()
        if resolved not in {"auto", "cpu", "cuda"}:
            raise ValueError(f"unsupported backend: {resolved}")
        if resolved == "auto":
            if cuda_available():
                resolved = "cuda"
            else:
                raise RuntimeError("auto backend could not find CUDA; pass --backend cpu explicitly for legacy CPU debugging")
        if resolved == "cuda" and not cuda_available():
            raise RuntimeError("CUDA backend requested for quad_ocean_monument, but nvcc/CUDA helper is not available")
        return quad_monument_prefilter_command(config, resolved, start, count, out_csv, force_build=force_build)
    raise ValueError(f"unsupported prefilter type: {kind}")


def prefilter_empty_header(config: Dict[str, Any]) -> str:
    kind = prefilter_type(config)
    if kind == "quad_ocean_monument":
        return (
            "seed,monument1_center_x,monument1_center_z,monument2_center_x,monument2_center_z,"
            "monument3_center_x,monument3_center_z,monument4_center_x,monument4_center_z,"
            "quad_window_chunk_x,quad_window_chunk_z,quad_window_min_block_x,quad_window_min_block_z,"
            "quad_window_max_block_x,quad_window_max_block_z,cluster_window_chunks,search_radius_blocks,"
            "mc_approx,monument_footprint_blocks,monument_center_offset_blocks,quad_backend,biome_viability_checked\n"
        )
    return "seed,best_slime_chunks,best_center_chunk_x,best_center_chunk_z,center_samples,circle_radius_chunks,search_radius_chunks\n"


def run_slime_prefilter(
    config: Dict[str, Any],
    *,
    start: int | None,
    count: int | None,
    append: bool,
    force_build: bool,
    backend: str = "cuda",
) -> Path:
    seed_range = config.get("seed_range", {})
    pf = config.get("prefilter", {})
    out = config.get("output", {})

    start = int(seed_range.get("start", 0) if start is None else start)
    count = int(seed_range.get("count", 1000000) if count is None else count)
    requested_backend = backend or str(pf.get("backend", "auto"))
    backend = resolve_slime_backend(requested_backend)

    odir = output_dir(config)
    csv_path = odir / out.get("candidates_csv", "candidates_slime.csv")
    cmd = slime_prefilter_command(config, backend, start, count, csv_path, force_build=force_build)
    if append:
        cmd.append("--append")
    print(f"[backend] requested={requested_backend} using={backend}", file=sys.stderr)
    print("[run]", " ".join(cmd), file=sys.stderr)
    t0 = time.monotonic()
    subprocess.run(cmd, check=True)
    elapsed = time.monotonic() - t0
    print(f"[done] slime prefilter wrote {csv_path} in {elapsed:.1f}s", file=sys.stderr)
    write_checkpoint(config, {
        "last_mode": "slime-prefilter",
        "backend": backend,
        "requested_backend": requested_backend,
        "start": start,
        "count": count,
        "elapsed_seconds": elapsed,
        "seeds_per_second": count / elapsed if elapsed > 0 else None,
        "output": str(csv_path),
    })
    return csv_path


def run_prefilter(
    config: Dict[str, Any],
    *,
    start: int | None,
    count: int | None,
    force_build: bool,
    backend: str = "cuda",
    out_csv: Optional[Path] = None,
) -> Path:
    seed_range = config.get("seed_range", {})
    out = config.get("output", {})
    start = int(seed_range.get("start", 0) if start is None else start)
    count = int(seed_range.get("count", 1000000) if count is None else count)
    if out_csv is None:
        out_csv = output_dir(config) / out.get("candidates_csv", "candidates.csv")
    cmd = prefilter_command(config, backend, start, count, out_csv, force_build=force_build)
    print(f"[prefilter] type={prefilter_type(config)}", file=sys.stderr)
    print("[run]", " ".join(cmd), file=sys.stderr)
    t0 = time.monotonic()
    subprocess.run(cmd, check=True)
    elapsed = time.monotonic() - t0
    print(f"[done] prefilter wrote {out_csv} in {elapsed:.1f}s", file=sys.stderr)
    write_checkpoint(config, {
        "last_mode": "prefilter",
        "prefilter_type": prefilter_type(config),
        "backend": backend,
        "start": start,
        "count": count,
        "elapsed_seconds": elapsed,
        "seeds_per_second": count / elapsed if elapsed > 0 else None,
        "output": str(out_csv),
    })
    return out_csv


def run_slime_window_filter(
    config: Dict[str, Any],
    candidates_csv: Path,
    *,
    run_dir: Optional[Path] = None,
    force_build: bool = False,
    limit: Optional[int] = None,
    backend: str = "cuda",
) -> Path:
    cfg = stage_config(config, "slime_window_density")
    resolved = (backend or "auto").lower()
    if resolved not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"unsupported backend: {resolved}")
    if resolved == "auto":
        if shutil.which("nvcc") is not None:
            resolved = "cuda"
        else:
            raise RuntimeError("auto backend could not find CUDA; pass --backend cpu explicitly for legacy CPU debugging")
    out_csv = output_file(config, "slime_windows_csv", "candidates_slime_windows.csv", run_dir)
    helper = compile_slime_window_cuda_helper(force=force_build) if resolved == "cuda" else compile_slime_window_helper(force=force_build)
    cmd = [
        str(helper),
        "--in", str(candidates_csv),
        "--out", str(out_csv),
        "--radius-blocks", str(int(cfg.get("search_radius_blocks", cfg.get("radius_blocks", 60000)))),
        "--window-chunks", str(int(cfg.get("window_chunks", 17))),
        "--min-slime-chunks", str(int(cfg.get("min_slime_chunks", cfg.get("threshold_chunks", 0)))),
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if resolved == "cuda":
        cuda_cfg = cfg.get("cuda", {})
        cmd += [
            "--threads", str(int(cuda_cfg.get("threads", 256))),
            "--blocks", str(int(cuda_cfg.get("blocks", 0))),
        ]
    print("[run]", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)
    return out_csv


def run_all_biomes_filter(
    config: Dict[str, Any],
    candidates_csv: Path,
    *,
    run_dir: Optional[Path] = None,
    force_build: bool = False,
    limit: Optional[int] = None,
    backend: str = "cuda",
) -> Path:
    cfg = stage_config(config, "all_biomes_within_origin_radius")
    requested = (backend or "auto").lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"unsupported backend: {requested}")
    if requested == "auto":
        if shutil.which("nvcc") is not None:
            requested = "cuda"
        else:
            raise RuntimeError("auto backend could not find CUDA; pass --backend cpu explicitly for legacy CPU verification")
    out_csv = output_file(config, "results_csv", "results_all_biomes.csv", run_dir)
    helper = compile_all_biomes_cuda_helper(force=force_build) if requested == "cuda" else compile_all_biomes_helper(config, force=force_build)
    cmd = [
        str(helper),
        "--in", str(candidates_csv),
        "--out", str(out_csv),
        "--radius-blocks", str(int(cfg.get("search_radius_blocks", cfg.get("radius_blocks", 10000)))),
        "--scale", str(int(cfg.get("scale", 64))),
        "--mc", minecraft_version_arg(config, cfg),
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if requested == "cuda":
        cuda_cfg = cfg.get("cuda", {})
        cmd += [
            "--threads", str(int(cuda_cfg.get("threads", 256))),
            "--batch-size", str(int(cuda_cfg.get("batch_size", 512))),
        ]
    print("[run]", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)
    return out_csv


def run_configured_postfilters(
    config: Dict[str, Any],
    candidates_csv: Path,
    *,
    run_dir: Optional[Path] = None,
    force_build: bool = False,
    limit: Optional[int] = None,
    backend: str = "cuda",
) -> Dict[str, str]:
    current = candidates_csv
    outputs: Dict[str, str] = {}
    order = config.get("pipeline", {}).get("order", [])
    first_stage = prefilter_type(config)
    for stage in order:
        if stage == first_stage or not stage_enabled(config, stage):
            continue
        if stage == "slime_window_density":
            current = run_slime_window_filter(config, current, run_dir=run_dir, force_build=force_build, limit=limit, backend=backend)
        elif stage == "all_biomes_within_origin_radius":
            current = run_all_biomes_filter(config, current, run_dir=run_dir, force_build=force_build, limit=limit, backend=backend)
        else:
            raise ValueError(f"unsupported configured pipeline stage: {stage}")
        outputs[stage] = str(current)
    return outputs


def run_configured_pipeline(
    config: Dict[str, Any],
    *,
    start: Optional[int],
    count: Optional[int],
    backend: str,
    force_build: bool,
    limit: Optional[int] = None,
) -> Dict[str, str]:
    prefilter_csv = run_prefilter(config, start=start, count=count, force_build=force_build, backend=backend)
    outputs = {"prefilter": str(prefilter_csv)}
    outputs.update(run_configured_postfilters(config, prefilter_csv, force_build=force_build, limit=limit, backend=backend))
    return outputs


def write_checkpoint(config: Dict[str, Any], data: Dict[str, Any]) -> None:
    out = config.get("output", {})
    cp = output_dir(config) / out.get("checkpoint_file", "checkpoint.json")
    data = {**data, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    cp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_external_checker(seed: int, check_name: str, check_config: Dict[str, Any], global_config: Dict[str, Any]) -> Dict[str, Any]:
    """Call a configured external checker.

    The checker should read one JSON object from stdin and print one JSON object:
      input:  {"seed": 123, "edition": "java", "version": "26.2", "fallback_version": "26.2", "biome_reference": "26.2", "check": "...", "config": {...}}
      output: {"ok": true, "details": {...}}
    """
    ext = check_config.get("external_checker", {})
    command = ext.get("command")
    if not command:
        return {"ok": None, "skipped": True, "reason": f"no external checker configured for {check_name}"}
    payload = {
        "seed": seed,
        "edition": global_config.get("edition", "java"),
        "version": global_config.get("target_version", "26.2"),
        "fallback_version": global_config.get("fallback_version", "26.2"),
        "biome_reference": global_config.get("biome_reference", "26.2"),
        "check": check_name,
        "config": check_config,
    }
    proc = subprocess.run(
        command if isinstance(command, list) else [command],
        input=json.dumps(payload, ensure_ascii=False),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return {"ok": False, "error": proc.stderr.strip() or proc.stdout.strip(), "returncode": proc.returncode}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"ok": False, "error": "checker did not return JSON", "stdout": proc.stdout[:1000]}


def refine_candidates(config: Dict[str, Any], candidates_csv: Path | None = None, limit: int | None = None) -> Path:
    odir = output_dir(config)
    out_cfg = config.get("output", {})
    if candidates_csv is None:
        candidates_csv = odir / out_cfg.get("candidates_csv", "candidates_slime.csv")
    results_csv = odir / out_cfg.get("results_csv", "results.csv")
    checks = config.get("checks", {})
    order = config.get("pipeline", {}).get("order", list(checks.keys()))

    with candidates_csv.open("r", encoding="utf-8", newline="") as src, results_csv.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.DictReader(src)
        fields = list(reader.fieldnames or []) + ["refine_status", "failed_check", "details_json"]
        writer = csv.DictWriter(dst, fieldnames=fields)
        writer.writeheader()
        n = 0
        for row in reader:
            if limit is not None and n >= limit:
                break
            n += 1
            seed = int(row["seed"])
            failed = ""
            details: Dict[str, Any] = {}
            status = "passed_configured_checks"
            for check_name in order:
                check_cfg = checks.get(check_name, {})
                if not check_cfg.get("enabled", True):
                    continue
                if check_name == "dense_slime_chunks":
                    # The prefilter already checked a stricter or equal condition for sampled centers.
                    details[check_name] = {"ok": True, "note": "covered by slime prefilter candidate row"}
                    continue
                result = run_external_checker(seed, check_name, check_cfg, config)
                details[check_name] = result
                if result.get("skipped"):
                    status = "pending_external_checkers"
                    continue
                if result.get("ok") is not True:
                    failed = check_name
                    status = "failed"
                    break
            row.update({"refine_status": status, "failed_check": failed, "details_json": json.dumps(details, ensure_ascii=False)})
            writer.writerow(row)
    print(f"[done] refine wrote {results_csv}", file=sys.stderr)
    return results_csv


def summarize_csv(path: Path, top: int = 10) -> None:
    if not path.exists():
        print(f"missing: {path}")
        return
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"{path}: {len(rows)} rows")
    for row in rows[:top]:
        print(row)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Minecraft Java 26.2 seed search framework")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="compile helper binaries")
    p_build.add_argument("--force", action="store_true")
    p_build.add_argument("--backend", choices=["auto", "cpu", "cuda"], default="cuda")
    p_build.add_argument("--target", choices=["prefilter", "slime", "monument", "slime-window", "biomes", "all"], default="prefilter")

    p_prefilter = sub.add_parser("prefilter", help="run configured first-stage prefilter")
    p_prefilter.add_argument("--start", type=int)
    p_prefilter.add_argument("--count", type=int)
    p_prefilter.add_argument("--force-build", action="store_true")
    p_prefilter.add_argument("--backend", choices=["auto", "cpu", "cuda"], default="cuda")

    p_slime = sub.add_parser("slime-prefilter", help="run fast sampled slime-density coarse filter")
    p_slime.add_argument("--start", type=int)
    p_slime.add_argument("--count", type=int)
    p_slime.add_argument("--append", action="store_true")
    p_slime.add_argument("--force-build", action="store_true")
    p_slime.add_argument("--backend", choices=["auto", "cpu", "cuda"], default="cuda", help="default uses CUDA; pass cpu only for legacy debugging")

    p_slime_window = sub.add_parser("slime-window", help="append exact 17x17 slime-window maximum for candidate seeds")
    p_slime_window.add_argument("--candidates", type=Path, required=True)
    p_slime_window.add_argument("--force-build", action="store_true")
    p_slime_window.add_argument("--limit", type=int)
    p_slime_window.add_argument("--backend", choices=["auto", "cpu", "cuda"], default="cuda")

    p_biomes = sub.add_parser("all-biomes", help="filter candidate seeds for all overworld biomes")
    p_biomes.add_argument("--candidates", type=Path, required=True)
    p_biomes.add_argument("--force-build", action="store_true")
    p_biomes.add_argument("--limit", type=int)
    p_biomes.add_argument("--backend", choices=["auto", "cpu", "cuda"], default="cuda")

    p_pipeline = sub.add_parser("pipeline", help="run configured prefilter and postfilters")
    p_pipeline.add_argument("--start", type=int)
    p_pipeline.add_argument("--count", type=int)
    p_pipeline.add_argument("--backend", choices=["auto", "cpu", "cuda"], default="cuda")
    p_pipeline.add_argument("--force-build", action="store_true")
    p_pipeline.add_argument("--limit", type=int, help="optional per-postfilter row limit for smoke tests")

    p_refine = sub.add_parser("refine", help="run configured external checkers over candidate CSV")
    p_refine.add_argument("--candidates", type=Path)
    p_refine.add_argument("--limit", type=int)

    p_sum = sub.add_parser("summarize", help="show candidate/result CSV summary")
    p_sum.add_argument("path", type=Path)
    p_sum.add_argument("--top", type=int, default=10)

    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    try:
        if args.cmd == "build":
            built: List[Path] = []
            target = args.target
            if target in {"prefilter", "all"}:
                if prefilter_type(cfg) == "quad_ocean_monument":
                    if args.backend == "cuda" or (args.backend == "auto" and shutil.which("nvcc") is not None):
                        built.append(compile_ocean_monument_cuda_helper(force=args.force))
                    else:
                        built.append(compile_ocean_monument_helper(cfg, force=args.force))
                else:
                    backend = resolve_slime_backend(args.backend)
                    built.append(compile_cuda_helper(force=args.force) if backend == "cuda" else compile_c_helper(force=args.force))
            if target in {"slime", "all"}:
                backend = resolve_slime_backend(args.backend)
                built.append(compile_cuda_helper(force=args.force) if backend == "cuda" else compile_c_helper(force=args.force))
            if target in {"monument", "all"} and not (target == "all" and prefilter_type(cfg) == "quad_ocean_monument"):
                if args.backend == "cuda" or (args.backend == "auto" and shutil.which("nvcc") is not None):
                    built.append(compile_ocean_monument_cuda_helper(force=args.force))
                else:
                    built.append(compile_ocean_monument_helper(cfg, force=args.force))
            if target in {"slime-window", "all"}:
                if args.backend == "cuda" or (args.backend == "auto" and shutil.which("nvcc") is not None):
                    built.append(compile_slime_window_cuda_helper(force=args.force))
                else:
                    built.append(compile_slime_window_helper(force=args.force))
            if target in {"biomes", "all"}:
                if args.backend == "cuda" or (args.backend == "auto" and shutil.which("nvcc") is not None):
                    built.append(compile_all_biomes_cuda_helper(force=args.force))
                else:
                    built.append(compile_all_biomes_helper(cfg, force=args.force))
            for path in built:
                print(path)
        elif args.cmd == "prefilter":
            print(run_prefilter(cfg, start=args.start, count=args.count, force_build=args.force_build, backend=args.backend))
        elif args.cmd == "slime-prefilter":
            print(run_slime_prefilter(cfg, start=args.start, count=args.count, append=args.append, force_build=args.force_build, backend=args.backend))
        elif args.cmd == "slime-window":
            print(run_slime_window_filter(cfg, args.candidates, force_build=args.force_build, limit=args.limit, backend=args.backend))
        elif args.cmd == "all-biomes":
            print(run_all_biomes_filter(cfg, args.candidates, force_build=args.force_build, limit=args.limit, backend=args.backend))
        elif args.cmd == "pipeline":
            print(json.dumps(run_configured_pipeline(
                cfg,
                start=args.start,
                count=args.count,
                backend=args.backend,
                force_build=args.force_build,
                limit=args.limit,
            ), ensure_ascii=False, indent=2))
        elif args.cmd == "refine":
            print(refine_candidates(cfg, candidates_csv=args.candidates, limit=args.limit))
        elif args.cmd == "summarize":
            summarize_csv(args.path, top=args.top)
    except (FileNotFoundError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
