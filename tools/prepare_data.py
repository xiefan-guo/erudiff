"""Build EruDiff training JSONL from the WISE benchmark prompt / rewrite pairs.

WISE ships, per category, two parallel files aligned on `prompt_id`: the
implicit prompt file and its `_rewrite` counterpart, which names the answer
directly. That is exactly the (implicit, explicit) structure EruDiff trains on:

  implicit  "Traditional food of the Mid-Autumn Festival"
  explicit  "Mooncakes."

`Explanation` is not used as y_expl, since it is a justification sentence rather
than an image-centric prompt; it is carried through as metadata only. y_found is
extracted from the implicit prompt.

Usage:
  python tools/download_wise.py
  python tools/prepare_data.py --output data/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.found_prompts import extract_found_prompts, extract_found_prompts_llm

# (implicit file, explicit/rewrite file, short category tag)
CATEGORIES = [
    ("cultural_common_sense.json",
     "cultural_common_sense_rewrite.json", "cultural"),
    ("natural_science.json",
     "natural_science_rewrite.json", "science"),
    ("spatio-temporal_reasoning.json",
     "spatio-temporal_reasoning_rewrite.json", "spatiotemporal"),
]


def _load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _clean_explicit(text: str) -> str:
    """Strip the trailing period WISE adds: "Mooncakes." -> "Mooncakes"."""
    text = text.strip()
    while text.endswith((".", " ")):
        text = text[:-1].strip()
    return text


def _jaccard(a: str, b: str) -> float:
    """Token-level Jaccard overlap between two prompts."""
    ta = {w.lower().strip(".,'\"()") for w in a.split() if w.strip(".,'\"()")}
    tb = {w.lower().strip(".,'\"()") for w in b.split() if w.strip(".,'\"()")}
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / max(len(ta | tb), 1)


def build_records(data_dir: Path, extractor, max_overlap: float | None = None,
                  ) -> tuple[list[dict], dict]:
    """Return (records, stats), skipping pairs that cannot be aligned.

    `max_overlap`, when set, drops pairs whose implicit and explicit prompts
    overlap at or above that Jaccard threshold. Such pairs are paraphrases: both
    prompts induce nearly the same distribution, so the DK-DM gradient in Eq. 2
    is close to zero and the sample mostly burns a forward pass.
    """
    records: list[dict] = []
    stats = {
        "per_category": {},
        "skipped_missing_rewrite": 0,
        "skipped_empty": 0,
        "skipped_identical": 0,
        "skipped_paraphrase": 0,
        "no_found_prompts": 0,
    }

    for impl_file, expl_file, tag in CATEGORIES:
        impl_path, expl_path = data_dir / impl_file, data_dir / expl_file
        if not impl_path.exists() or not expl_path.exists():
            print(f"  ! missing {impl_file} or {expl_file}, skipping category")
            continue

        implicit_items = _load(impl_path)
        explicit_by_id = {
            str(it.get("prompt_id", it.get("id", ""))): it
            for it in _load(expl_path)
        }

        kept = 0
        for item in implicit_items:
            pid = str(item.get("prompt_id", item.get("id", "")))
            implicit = (item.get("Prompt") or "").strip()
            counterpart = explicit_by_id.get(pid)

            if counterpart is None:
                stats["skipped_missing_rewrite"] += 1
                continue

            explicit = _clean_explicit(counterpart.get("Prompt") or "")
            if not implicit or not explicit:
                stats["skipped_empty"] += 1
                continue

            # An identical pair carries no implicit -> explicit signal.
            if implicit.strip().lower() == explicit.strip().lower():
                stats["skipped_identical"] += 1
                continue

            overlap = _jaccard(implicit, explicit)
            if max_overlap is not None and overlap >= max_overlap:
                stats["skipped_paraphrase"] += 1
                continue

            found = extractor(implicit)
            if not found:
                stats["no_found_prompts"] += 1

            records.append({
                "id": f"wise_{tag}_{pid}",
                "prompt": implicit,            # y_impl
                "aug_prompt": explicit,        # y_expl
                "sub_prompt": found,           # y_found
                "category": item.get("Category", ""),
                "subcategory": item.get("Subcategory", ""),
                # Lexical divergence of the pair; low means a strong knowledge gap.
                "overlap": round(overlap, 4),
                "explanation": (counterpart.get("Explanation") or "").strip(),
            })
            kept += 1

        stats["per_category"][tag] = kept
        print(f"  {tag:15s} {kept:4d} pairs")

    return records, stats


def report(records: list[dict], stats: dict) -> None:
    print(f"\nusable (implicit, explicit) pairs: {len(records)}")
    print(f"  no matching rewrite id : {stats['skipped_missing_rewrite']}")
    print(f"  empty prompt field     : {stats['skipped_empty']}")
    print(f"  implicit == explicit   : {stats['skipped_identical']}")
    print(f"  paraphrase (filtered)  : {stats['skipped_paraphrase']}")
    print(f"  kept without y_found   : {stats['no_found_prompts']}")

    if not records:
        return

    overlaps = [r["overlap"] for r in records]
    print(f"\nmean implicit/explicit overlap: {sum(overlaps) / len(overlaps):.3f}")
    print("\nexample:")
    r = records[0]
    print(f"  y_impl  : {r['prompt']}")
    print(f"  y_expl  : {r['aug_prompt']}")
    print(f"  y_found : {r['sub_prompt']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(ROOT / "data"),
                        help="directory holding the downloaded WISE json files")
    parser.add_argument("--output", default=str(ROOT / "data" / "train.jsonl"))
    parser.add_argument("--found-extractor", default="rule",
                        choices=["rule", "llm"],
                        help="y_found extractor; 'llm' requires implementing "
                             "extract_found_prompts_llm")
    parser.add_argument("--max-overlap", type=float, default=None,
                        help="drop pairs whose implicit/explicit Jaccard overlap "
                             "is >= this value (e.g. 0.5 removes paraphrases). "
                             "Default: keep all pairs.")
    parser.add_argument("--dry-run", action="store_true",
                        help="report statistics without writing output")
    args = parser.parse_args()

    extractor = (extract_found_prompts if args.found_extractor == "rule"
                 else extract_found_prompts_llm)

    print("building (implicit, explicit) pairs from WISE prompt/rewrite files:")
    records, stats = build_records(Path(args.data_dir), extractor,
                                   max_overlap=args.max_overlap)
    report(records, stats)

    if args.dry_run:
        print("\n(--dry-run: no file written)")
        return

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(records)} records -> {out}")


if __name__ == "__main__":
    main()
