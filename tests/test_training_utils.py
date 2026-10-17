"""Checks for the training strategies: TA-CL (Eq. 3), AF-KC, EMAN, D_loss filtering.

Run: python -m pytest tests/test_training_utils.py
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.utils.training import (
    EMANormalizer,
    FailureFilter,
    PromptMixer,
    TimestepCurriculum,
)


def test_curriculum_distribution():
    cur = TimestepCurriculum(num_inference_steps=30, decay=0.1, seed=0)
    assert abs(sum(cur.probs) - 1.0) < 1e-9
    assert all(cur.probs[i] > cur.probs[i + 1] for i in range(len(cur.probs) - 1))
    # Closed form: p(n) / p(n+1) = exp(lambda).
    ratio = cur.probability(1) / cur.probability(2)
    assert abs(ratio - math.exp(0.1)) < 1e-9
    # Early steps dominate: the first third carries more than a third of the mass.
    assert sum(cur.probs[:10]) > 0.40


def test_curriculum_sampling():
    cur = TimestepCurriculum(num_inference_steps=30, decay=0.1, seed=0)
    draws = Counter(cur.sample() for _ in range(40000))
    assert all(1 <= n <= 30 for n in draws)
    assert abs(draws[1] / 40000 - cur.probability(1)) < 0.01


def test_curriculum_zero_decay_is_uniform():
    flat = TimestepCurriculum(30, decay=0.0, seed=0)
    assert all(abs(p - 1 / 30) < 1e-12 for p in flat.probs)


def test_mixer_rates():
    mixer = PromptMixer(seed=0)
    modes = Counter(mixer.sample_mode() for _ in range(60000))
    for mode, want in (("impl", 0.8), ("expl", 0.1), ("found", 0.1)):
        assert abs(modes[mode] / 60000 - want) < 0.01


def test_mixer_pairs():
    mixer = PromptMixer(seed=0)
    assert mixer.resolve("impl", "IMPL", "EXPL", "FOUND") == ("IMPL", "EXPL")
    assert mixer.resolve("expl", "IMPL", "EXPL", "FOUND") == ("EXPL", "EXPL")
    assert mixer.resolve("found", "IMPL", "EXPL", "FOUND") == ("FOUND", "FOUND")
    # Without a y_found the term degrades to explicit consolidation.
    assert mixer.resolve("found", "IMPL", "EXPL", None) == ("EXPL", "EXPL")


def test_mixer_rejects_bad_probabilities():
    try:
        PromptMixer(p_impl=0.5, p_expl=0.1, p_found=0.1)
    except ValueError:
        return
    raise AssertionError("probabilities that do not sum to 1 must be rejected")


def test_eman_balances_scales():
    ema = EMANormalizer(decay=0.99)
    first = ema.normalize("dkdm", torch.tensor(5.0))
    assert abs(first.item() - 1.0) < 1e-6
    for _ in range(200):
        ema.normalize("dkdm", torch.tensor(5.0))
    assert abs(ema.get("dkdm") - 5.0) < 1e-3

    # Two losses three orders of magnitude apart both land near 1.
    ema2 = EMANormalizer(decay=0.99)
    for _ in range(500):
        big = ema2.normalize("big", torch.tensor(1000.0))
        small = ema2.normalize("small", torch.tensor(0.001))
    assert abs(big.item() - 1.0) < 0.01 and abs(small.item() - 1.0) < 0.01


def test_eman_preserves_gradient():
    x = torch.tensor([2.0], requires_grad=True)
    ema = EMANormalizer(decay=0.99)
    ema.normalize("k", torch.tensor(4.0))
    ema.normalize("k", (x * 2).sum()).backward()
    assert x.grad is not None and x.grad.item() > 0


def test_failure_filter_fixed_threshold():
    flt = FailureFilter(threshold=23.31)
    mask = flt.is_failure(torch.tensor([20.0, 23.31, 25.0, 10.0]))
    assert mask.tolist() == [True, False, False, True]
    assert flt.effective_threshold == 23.31


def test_failure_filter_online_calibration():
    flt = FailureFilter(threshold=None, warmup=10)
    # Nothing is marked before calibration, so NO-RL stays inert.
    assert not flt.is_failure(torch.tensor([1.0, 2.0])).any()
    flt.observe([10.0] * 5 + [20.0] * 5)
    assert abs(flt.effective_threshold - 15.0) < 1e-9
    assert flt.is_failure(torch.tensor([9.0, 21.0])).tolist() == [True, False]
