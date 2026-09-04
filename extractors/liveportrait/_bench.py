"""Bench LP extractor batch_size × num_workers on 1080p videos.

1080p is benched because it is the heavy-corpus realistic workload
(1080p 10-bit H.264, 6800 frames per video). Picks a small subset
(default 5 videos) and runs each combination. Reports throughput in fps.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
# repo-relative defaults: the corpus lives under data/ (see README); override for other 1080p sets
_DATA_GT = Path(__file__).resolve().parents[2] / "data" / "GT"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--videos_dir", type=Path, default=_DATA_GT / "LS_AL",
                   help="dir of 1080p .mp4 to bench on (default: repo data/GT/LS_AL)")
    p.add_argument("--insightface_dir", type=Path, default=_DATA_GT / "LS_AL_insightface",
                   help="cached lm106 npzs for --videos_dir (default: repo data/GT/LS_AL_insightface)")
    p.add_argument("--n_videos", type=int, default=5,
                   help="how many videos to bench on")
    p.add_argument("--batches",  default="16,32,64",
                   help="comma-separated batch sizes")
    p.add_argument("--workers",  default="1,2,4",
                   help="comma-separated num_workers")
    p.add_argument("--scratch",  type=Path, default=Path("/tmp/lp_bench"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.videos_dir.is_dir():
        raise SystemExit(f"--videos_dir not found: {args.videos_dir}\n"
                         "Point it at a directory of 1080p .mp4 (default: repo data/GT/LS_AL).")
    if not args.insightface_dir.is_dir():
        raise SystemExit(f"--insightface_dir not found: {args.insightface_dir}\n"
                         "It must hold the cached lm106 npzs for --videos_dir.")
    if args.scratch.exists():
        shutil.rmtree(args.scratch)
    args.scratch.mkdir(parents=True)
    in_dir = args.scratch / "videos"
    in_dir.mkdir()

    bench_videos = sorted(args.videos_dir.glob("*.mp4"))[: args.n_videos]
    print(f"using {len(bench_videos)} videos:")
    for v in bench_videos:
        shutil.copy(v, in_dir / v.name)
        print(f"  {v.name}")

    batches = [int(b) for b in args.batches.split(",")]
    workers = [int(w) for w in args.workers.split(",")]

    print(f"\nbench grid: batch ∈ {batches}, workers ∈ {workers}")
    print()
    print(f"{'batch':>6} {'workers':>8}  {'wall_sec':>10} {'fps_videos':>12}")
    print("-" * 44)

    results = []
    for bs in batches:
        for nw in workers:
            out_dir = args.scratch / f"out_b{bs}_w{nw}"
            if out_dir.exists():
                shutil.rmtree(out_dir)
            t0 = time.perf_counter()
            cmd = [
                "pixi", "run", "python", "extract.py",
                "--videos_dir",      str(in_dir),
                "--insightface_dir", str(args.insightface_dir),
                "--out_dir",         str(out_dir),
                "--batch_size",      str(bs),
                "--num_workers",     str(nw),
                "--overwrite",
            ]
            r = subprocess.run(cmd, cwd=str(REPO),
                               capture_output=True, text=True)
            wall = time.perf_counter() - t0
            ok = (r.returncode == 0)
            fps = (len(bench_videos) / wall) if ok else 0.0
            print(f"{bs:>6} {nw:>8}  {wall:>10.1f} {fps:>12.4f}",
                  "" if ok else "  FAIL")
            if not ok:
                print(r.stderr.splitlines()[-3:])
            results.append({"batch": bs, "workers": nw,
                            "wall": wall, "fps_videos": fps, "ok": ok})

    # Best
    print()
    ok_results = [r for r in results if r["ok"]]
    if ok_results:
        best = min(ok_results, key=lambda r: r["wall"])
        print(f"best:  batch={best['batch']}  workers={best['workers']}  wall={best['wall']:.1f}s")


if __name__ == "__main__":
    main()
