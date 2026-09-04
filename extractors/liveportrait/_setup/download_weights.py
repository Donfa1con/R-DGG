"""Pull LivePortrait motion_extractor.pth from the official HF mirror.

We only need the motion_extractor weights — the rest of the LP pipeline
(appearance feature, warping, generator, stitching) is for *generating*
video; we only consume the encoder side.
"""
from __future__ import annotations

import sys
from pathlib import Path
from urllib.request import urlretrieve

WEIGHTS = {
    # KwaiVGI publishes the official weights on HuggingFace.
    "motion_extractor.pth":
        "https://huggingface.co/KwaiVGI/LivePortrait/resolve/main/liveportrait/base_models/motion_extractor.pth",
}

DEST_DIR = Path(__file__).resolve().parent.parent / "_weights"


def main() -> int:
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in WEIGHTS.items():
        out = DEST_DIR / name
        if out.is_file() and out.stat().st_size > 1_000_000:
            print(f"  skip {name} (already {out.stat().st_size:,} bytes)")
            continue
        print(f"  download {name} from {url}")
        try:
            urlretrieve(url, out)
            print(f"  ok: {out} ({out.stat().st_size:,} bytes)")
        except Exception as e:
            print(f"  FAIL {name}: {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
