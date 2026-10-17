"""Checks that the prompt-triplet JSONL loader matches the documented format.

Run: python -m pytest tests/test_dataset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import PromptTripletDataset

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample_train.jsonl"


def test_sample_file_loads():
    dataset = PromptTripletDataset(SAMPLE, seed=0)
    assert len(dataset) > 0
    assert dataset.coverage["total"] == len(dataset)

    item = dataset[0]
    assert set(item) == {"id", "impl", "expl", "found"}
    assert item["impl"] and item["expl"]


def test_found_candidates_rotate(tmp_path):
    path = tmp_path / "triplets.jsonl"
    path.write_text(json.dumps({
        "id": "x",
        "prompt": "implicit",
        "aug_prompt": "explicit",
        "sub_prompt": ["first", "second", "third"],
    }) + "\n", encoding="utf-8")

    dataset = PromptTripletDataset(path, seed=0)
    drawn = {dataset[0]["found"] for _ in range(50)}
    assert drawn == {"first", "second", "third"}


def test_missing_fields_rejected(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"id": "x", "prompt": "implicit"}) + "\n",
                    encoding="utf-8")
    try:
        PromptTripletDataset(path)
    except ValueError as exc:
        assert "aug_prompt" in str(exc)
        return
    raise AssertionError("a record without aug_prompt must be rejected")


def test_samples_without_found_yield_empty_string(tmp_path):
    path = tmp_path / "nofound.jsonl"
    path.write_text(json.dumps({
        "id": "x", "prompt": "implicit", "aug_prompt": "explicit",
    }) + "\n", encoding="utf-8")

    dataset = PromptTripletDataset(path)
    assert dataset[0]["found"] == ""
    assert dataset.coverage["without_found"] == 1
