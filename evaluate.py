"""CLIP-based knowledge-alignment evaluation for generated image sets.

Scores how well each image matches its explicit anchor,

    knowledge_score = CLIP(image, y_expl)

and reports the paired preference against a baseline run,

    win_rate = P[ CLIP(image_erudiff, y_expl) > CLIP(image_baseline, y_expl) ].

This is a proxy, not WiScore: the official metric is computed by a judge model
rating Consistency, Realism and Aesthetic Quality, so absolute values here are
not comparable to published WiScore numbers. Use it to compare runs scored the
same way, and run the official WISE evaluator on `generate.py` output for true
WiScore.

Example:
    python evaluate.py --baseline out/baseline --candidate out/erudiff \
        --data data/sample_train.jsonl --clip openai/clip-vit-large-patch14
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image


def load_manifest(d: Path) -> dict[str, dict]:
    mf = d / "manifest.jsonl"
    if not mf.exists():
        raise FileNotFoundError(f"missing manifest: {mf}")
    out = {}
    with open(mf, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[rec["id"]] = rec
    return out


def load_pairs(path: Path) -> dict[str, dict]:
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[str(rec["id"])] = rec
    return out


class ClipScorer:
    """CLIP image-text similarity; works with any CLIP-architecture checkpoint."""

    def __init__(self, model_path: str, device: str = "cuda",
                 dtype: torch.dtype = torch.float32):
        from transformers import AutoModel, AutoProcessor
        self.device, self.dtype = device, dtype
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).eval().to(device, dtype)

    @torch.no_grad()
    def score(self, images: list[Image.Image], texts: list[str]) -> torch.Tensor:
        img_in = self.processor(images=images, return_tensors="pt").to(self.device)
        txt_in = self.processor(text=texts, padding=True, truncation=True,
                                max_length=77, return_tensors="pt")
        txt_in = {k: v.to(self.device) for k, v in txt_in.items()}

        # Feature getters return either a tensor or a ModelOutput depending on
        # the transformers version.
        def _to_tensor(out):
            return out.image_embeds if hasattr(out, "image_embeds") else (
                out.pooler_output if hasattr(out, "pooler_output") else out
            )

        ie = F.normalize(_to_tensor(self.model.get_image_features(**img_in)), dim=-1)
        te = F.normalize(_to_tensor(self.model.get_text_features(**txt_in)), dim=-1)
        sims = (ie * te).sum(dim=-1)
        return sims * self.model.logit_scale.exp()


def summarize(name: str, scores: dict[str, float],
              pairs: dict[str, dict]) -> dict:
    by_cat: dict[str, list[float]] = defaultdict(list)
    for sid, s in scores.items():
        by_cat[pairs.get(sid, {}).get("category") or "unknown"].append(s)
    vals = list(scores.values())
    return {
        "name": name,
        "n": len(vals),
        "mean": sum(vals) / len(vals) if vals else float("nan"),
        "per_category": {c: sum(v) / len(v) for c, v in sorted(by_cat.items())},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True, help="image dir (with manifest)")
    p.add_argument("--baseline", default=None,
                   help="optional second image dir for a paired comparison")
    p.add_argument("--data", required=True,
                   help="JSONL with prompt/aug_prompt per id")
    p.add_argument("--clip", required=True, help="CLIP-architecture checkpoint")
    p.add_argument("--target_field", default="aug_prompt",
                   choices=["aug_prompt", "prompt", "explanation"],
                   help="text to score against; aug_prompt (the explicit "
                        "anchor) is the knowledge-alignment target")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--out", default=None, help="write JSON results here")
    args = p.parse_args()

    pairs = load_pairs(Path(args.data))
    scorer = ClipScorer(args.clip)

    def score_dir(d: Path) -> dict[str, float]:
        manifest = load_manifest(d)
        ids = [i for i in manifest if i in pairs]
        missing = len(manifest) - len(ids)
        if missing:
            print(f"  ! {missing} images had no matching prompt record")
        out: dict[str, float] = {}
        for start in range(0, len(ids), args.batch_size):
            chunk = ids[start:start + args.batch_size]
            imgs = [Image.open(d / manifest[i]["file"]).convert("RGB") for i in chunk]
            texts = [pairs[i].get(args.target_field) or pairs[i]["aug_prompt"]
                     for i in chunk]
            for i, v in zip(chunk, scorer.score(imgs, texts).tolist()):
                out[i] = v
        return out

    print(f"scoring candidate: {args.candidate}")
    cand = score_dir(Path(args.candidate))
    results = {
        "metric": f"CLIP proxy (not WiScore), target={args.target_field}",
        "clip_model": args.clip,
        "candidate": summarize("candidate", cand, pairs),
    }

    if args.baseline:
        print(f"scoring baseline:  {args.baseline}")
        base = score_dir(Path(args.baseline))
        results["baseline"] = summarize("baseline", base, pairs)

        common = sorted(set(cand) & set(base))
        if common:
            wins = sum(1 for i in common if cand[i] > base[i])
            deltas = [cand[i] - base[i] for i in common]
            per_cat: dict[str, list[int]] = defaultdict(list)
            for i in common:
                cat = pairs.get(i, {}).get("category") or "unknown"
                per_cat[cat].append(1 if cand[i] > base[i] else 0)
            results["paired"] = {
                "n": len(common),
                "win_rate": wins / len(common),
                "mean_delta": sum(deltas) / len(deltas),
                "per_category_win_rate": {
                    c: sum(v) / len(v) for c, v in sorted(per_cat.items())
                },
            }

    print("\n" + "=" * 66)
    print(results["metric"])
    print("=" * 66)
    for key in ("baseline", "candidate"):
        if key in results:
            r = results[key]
            print(f"\n{r['name']:10s} n={r['n']}  mean={r['mean']:.4f}")
            for c, v in r["per_category"].items():
                print(f"  {c:28s} {v:.4f}")
    if "paired" in results:
        pr = results["paired"]
        print(f"\npaired (n={pr['n']}):")
        print(f"  win rate    {pr['win_rate'] * 100:.1f}%")
        print(f"  mean delta  {pr['mean_delta']:+.4f}")
        for c, v in pr["per_category_win_rate"].items():
            print(f"    {c:26s} {v * 100:.1f}%")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
