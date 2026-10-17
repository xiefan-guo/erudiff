"""Generate images from a prompt set, optionally with a trained EruDiff LoRA.

Used both for qualitative comparison (baseline vs. +EruDiff on the same seed)
and to produce the image set that `evaluate.py` scores.

Example -- baseline and adapted, same seeds:
    python generate.py --prompts data/sample_train.jsonl --field prompt \
        --out out/baseline
    python generate.py --prompts data/sample_train.jsonl --field prompt \
        --lora runs/erudiff/final --out out/erudiff
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import FluxPipeline


def load_prompts(path: Path, field: str, limit: int | None) -> list[dict]:
    """Read prompts from a .jsonl, .json or plain-text file."""
    items: list[dict] = []
    if path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    elif path.suffix == ".json":
        with open(path, encoding="utf-8") as fh:
            items = json.load(fh)
    else:
        with open(path, encoding="utf-8") as fh:
            items = [{"prompt": line.strip()} for line in fh if line.strip()]

    out = []
    for i, it in enumerate(items):
        text = it.get(field) or it.get("prompt") or it.get("Prompt")
        if not text:
            continue
        out.append({
            "id": str(it.get("id", it.get("prompt_id", i))),
            "prompt": text,
            "category": it.get("category", it.get("Category", "")),
            "subcategory": it.get("subcategory", it.get("Subcategory", "")),
        })
        if limit and len(out) >= limit:
            break
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="black-forest-labs/FLUX.1-dev",
                   help="repo id or local snapshot directory")
    p.add_argument("--lora", default=None, help="trained LoRA directory")
    p.add_argument("--lora_scale", type=float, default=1.0)
    p.add_argument("--prompts", required=True)
    p.add_argument("--field", default="prompt",
                   help="which field to render: 'prompt' (implicit) or "
                        "'aug_prompt' (explicit anchor)")
    p.add_argument("--out", required=True)
    p.add_argument("--num", type=int, default=None, help="limit prompt count")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--seed", type=int, default=42,
                   help="base seed; image k uses seed + k so runs are comparable")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = p.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(Path(args.prompts), args.field, args.num)
    print(f"{len(prompts)} prompts from {args.prompts} (field={args.field})")

    print(f"loading FLUX from {args.model} ...", flush=True)
    pipe = FluxPipeline.from_pretrained(args.model, dtype=dtype)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    if args.lora:
        print(f"loading LoRA from {args.lora} (scale={args.lora_scale})")
        pipe.load_lora_weights(args.lora, adapter_name="erudiff")
        pipe.set_adapters(["erudiff"], adapter_weights=[args.lora_scale])
    else:
        print("no LoRA: baseline FLUX.1-dev")

    manifest = []
    for start in range(0, len(prompts), args.batch_size):
        chunk = prompts[start:start + args.batch_size]
        gens = [torch.Generator("cuda").manual_seed(args.seed + start + k)
                for k in range(len(chunk))]
        images = pipe(
            prompt=[c["prompt"] for c in chunk],
            height=args.resolution,
            width=args.resolution,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=gens,
        ).images

        for c, img in zip(chunk, images):
            fname = f"{c['id']}.png"
            img.save(out_dir / fname)
            manifest.append({**c, "file": fname})

        print(f"  {min(start + args.batch_size, len(prompts))}/{len(prompts)}",
              flush=True)

    with open(out_dir / "manifest.jsonl", "w", encoding="utf-8") as fh:
        for m in manifest:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"wrote {len(manifest)} images -> {out_dir}")


if __name__ == "__main__":
    main()
