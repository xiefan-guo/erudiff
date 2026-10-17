## Refactoring Knowledge in Diffusion Models for Advanced T2I<br><sub>Official PyTorch code release for the ECCV 2026 paper: [https://arxiv.org/abs/2603.20828](https://arxiv.org/abs/2603.20828)</sub>

**EruDiff: Refactoring Knowledge in Diffusion Models for Advanced Text-to-Image Synthesis**<br>
Xiefan Guo, Xinzhu Ma, Haoxiang Ma, Zihao Zhou, Di Huang<br>

Abstract: *Text-to-image diffusion models have achieved remarkable fidelity in synthesizing images from explicit text prompts, yet exhibit a critical deficiency in processing implicit prompts that require deep-level world knowledge, ranging from natural sciences to cultural commonsense, resulting in counter-factual synthesis. This paper traces the root of this limitation to a fundamental dislocation of the underlying knowledge structures, manifesting as a chaotic organization of implicit prompts compared to their explicit counterparts. In this paper, we propose EruDiff, which aims to refactor the knowledge within diffusion models. Specifically, we develop the Diffusion Knowledge Distribution Matching (DK-DM) to register the knowledge distribution of intractable implicit prompts with that of well-defined explicit anchors. Furthermore, to rectify the inherent biases in explicit prompt rendering, we employ the Negative-Only Reinforcement Learning (NO-RL) strategy for fine-grained correction. Rigorous empirical evaluations demonstrate that our method significantly enhances the performance of leading diffusion models, including FLUX and Qwen-Image, across both the scientific knowledge benchmark (i.e., Science-T2I) and the world knowledge benchmark (i.e., WISE), underscoring the effectiveness and generalizability.*

This repository implements EruDiff on top of FLUX.1-dev with LoRA.

## Layout


```
train.py                   Phase I (DK-DM) + Phase II (NO-RL)
generate.py                image synthesis, with or without a trained LoRA
evaluate.py                CLIP-based knowledge-alignment scoring
src/utils/reward.py        CLIP-style reward model wrapper (NO-RL only)
src/data/dataset.py        (y_impl, y_expl, y_found) triplets
src/data/found_prompts.py  y_found extraction
tools/download_wise.py     fetch the WISE prompt/rewrite files
tools/prepare_data.py      build the training JSONL
scripts/                   single-GPU smoke run, multi-GPU training
tests/                     CPU unit tests
```

## Installation

```bash
conda create -n erudiff python=3.11 -y && conda activate erudiff
pip install -r requirements.txt
```

Verified with torch 2.7.1 (cu126), diffusers 0.40.0, transformers 5.17.0,
peft 0.21.1, accelerate 1.15.0. FLUX.1-dev is a gated checkpoint: accept its
licence on the Hugging Face Hub and log in with `huggingface-cli login`, or pass
a local snapshot directory to `--pretrained_model_name_or_path`.

## Data

Training needs only text: one JSONL record per sample.

```jsonl
{"id": "", "prompt": "", "aug_prompt": "", "sub_prompt": [""]}
```

| field | symbol | meaning |
|---|---|---|
| `prompt` | `y_impl` | implicit prompt requiring world knowledge |
| `aug_prompt` | `y_expl` | explicit anchor naming the answer directly |
| `sub_prompt` | `y_found` | short entity / proper-noun units used by AF-KC |

`data/sample_train.jsonl` holds 12 such records so the pipeline is runnable out
of the box. For a full corpus, the prompt/rewrite pairs of the
[WISE benchmark](https://github.com/PKU-YuanGroup/WISE) have the same structure
and can be converted directly:

```bash
python tools/download_wise.py                      # -> data/*.json
python tools/prepare_data.py --output data/train.jsonl
python tools/prepare_data.py --dry-run             # statistics only
```

Pairs that are mere paraphrases of each other give a near-zero DK-DM gradient;
drop them with `--max-overlap 0.5` (token-level Jaccard). The default keeps
every pair.

## Training

```bash
# single GPU, 6 steps, checks the loop end to end
FLUX=black-forest-labs/FLUX.1-dev bash scripts/smoke_train.sh

# full run
FLUX=black-forest-labs/FLUX.1-dev DATA=data/train.jsonl bash scripts/train.sh
GPUS=0,1 STEPS=4000 bash scripts/train.sh
```

NO-RL is off unless a reward model is given, in which case training uses DK-DM
alone. To enable it:

```bash
bash scripts/train.sh --reward_model_path <reward-model> --reward_threshold 23.31
```

Omitting `--reward_threshold` calibrates it online from the running mean of
observed scores.

Checkpoints are written as diffusers-format LoRA weights to
`$OUT/checkpoint-<step>/`, and the final adapter to `$OUT/final/`.

## Inference

```bash
python generate.py --prompts data/sample_train.jsonl --out out/baseline
python generate.py --prompts data/sample_train.jsonl --out out/erudiff \
    --lora runs/erudiff/final
```

Image `k` uses seed `--seed + k`, so two runs over the same prompt file are
directly comparable image by image.

## Evaluation

`evaluate.py` scores how well each image matches its explicit anchor with CLIP
and reports the paired win rate against a baseline run:

```bash
python evaluate.py --baseline out/baseline --candidate out/erudiff \
    --data data/train.jsonl --clip openai/clip-vit-large-patch14
```

This is a lightweight proxy, not WiScore, which is computed by a judge model
rating Consistency, Realism and Aesthetic Quality. Absolute values are therefore
not comparable to published WiScore numbers — use it only to compare runs scored
the same way, and run the official WISE evaluator on `generate.py` output for
true WiScore.

## Citation

```bibtex
@inproceedings{guo2026erudiff,
  title     = {EruDiff: Refactoring Knowledge in Diffusion Models for Advanced Text-to-Image Synthesis},
  author    = {Guo, Xiefan and Ma, Xinzhu and Ma, Haoxiang and Zhou, Zihao and Huang, Di},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## Acknowledgements

Built on [FLUX.1-dev](https://github.com/black-forest-labs/flux),
[diffusers](https://github.com/huggingface/diffusers) and
[peft](https://github.com/huggingface/peft). The training-data pipeline uses the
prompt/rewrite pairs of the [WISE](https://github.com/PKU-YuanGroup/WISE)
benchmark.
