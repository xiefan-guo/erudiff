"""Download the WISE benchmark prompt / rewrite files used for training data.

WISE ships, per category, two parallel JSON files aligned on `prompt_id`: the
implicit prompt file and its `_rewrite` counterpart holding the explicit answer.
Files are fetched through the GitHub contents API into `data/`.

Source: https://github.com/PKU-YuanGroup/WISE

Usage:
    python tools/download_wise.py [--out data]
"""

from __future__ import annotations

import argparse
import base64
import json
import urllib.request
from pathlib import Path

REPO = "PKU-YuanGroup/WISE"

FILES = [
    ("WISE_legacy/data/cultural_common_sense.json",
     "cultural_common_sense.json"),
    ("WISE_legacy/data/cultural_common_sense_rewrite.json",
     "cultural_common_sense_rewrite.json"),
    ("WISE_legacy/data/natural_science.json",
     "natural_science.json"),
    ("WISE_legacy/data/natural_science_rewrite.json",
     "natural_science_rewrite.json"),
    ("WISE_legacy/data/spatio-temporal_reasoning.json",
     "spatio-temporal_reasoning.json"),
    ("WISE_legacy/data/spatio-temporal_reasoning_rewrite.json",
     "spatio-temporal_reasoning_rewrite.json"),
]


def fetch(repo_path: str, dest: Path) -> None:
    url = f"https://api.github.com/repos/{REPO}/contents/{repo_path}"
    req = urllib.request.Request(url, headers={"User-Agent": "python"})
    with urllib.request.urlopen(req, timeout=60) as response:
        meta = json.loads(response.read())
    if "content" not in meta:
        raise RuntimeError(f"unexpected API response for {repo_path}")
    data = json.loads(base64.b64decode(meta["content"].replace("\n", "")))
    dest.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"  {dest.name}: {len(data)} entries")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data", help="output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"downloading WISE prompt files from {REPO}:")
    for repo_path, name in FILES:
        try:
            fetch(repo_path, out_dir / name)
        except Exception as exc:  # network / layout changes
            print(f"  ! {name} failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
