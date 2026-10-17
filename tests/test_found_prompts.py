"""Checks for y_found extraction against the examples the paper states.

Run: python -m pytest tests/test_found_prompts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.found_prompts import extract_found_prompts

# (prompt, units that must be extracted), taken from the paper.
PAPER_CASES = [
    ("The shield used by Captain America", {"The shield", "Captain America"}),
    ("The sword of King Arthur that he pulled from the stone",
     {"The sword", "King Arthur"}),
    ("Albert Einstein's favorite musical instruments",
     {"Albert Einstein", "musical instruments"}),
]

POLICY_CASES = [
    "A napkin in a simple space without gravity, simple, clear and realistic.",
    "A glass of juice at hundreds of degrees Celsius in the bedroom, realistic.",
    "Depict a glass of water at an arctic environment, highlighting water's state",
    "The most iconic architectural structure of ancient Egypt",
    "Traditional food of the Mid-Autumn Festival",
    "A monkey from China with a blue face and golden fur, adapted to cold climates",
    "The Sydney Opera House when it's 8 AM in San Francisco",
    "An apple tree with leaves exhibiting chlorophyll breakdown",
    "A structure that touches the sky, showcasing modern design and urban "
    "aspirations in the American city of New York",
]

# Words that must never appear in a kept unit, per the policy's exclusion list.
# Matched word-wise, since a substring test wrongly flags proper nouns.
FORBIDDEN_WORDS = {
    "realistic", "realism", "simple", "clear", "highlighting", "depict",
    "showcasing", "exhibiting", "is", "that", "which", "without", "favorite",
    "concept", "symbol", "moment", "scene",
}


def test_paper_examples():
    for prompt, required in PAPER_CASES:
        got = {unit.lower() for unit in extract_found_prompts(prompt)}
        missing = {r for r in required if r.lower() not in got}
        assert not missing, f"{prompt!r}: missing {sorted(missing)}, got {sorted(got)}"


def test_no_forbidden_content():
    for prompt in POLICY_CASES:
        for unit in extract_found_prompts(prompt):
            words = {w.lower().strip(".,'") for w in unit.split()}
            assert not FORBIDDEN_WORDS & words, f"{prompt!r}: forbidden in {unit!r}"


def test_units_stay_short():
    for prompt in POLICY_CASES:
        for unit in extract_found_prompts(prompt):
            assert 1 <= len(unit.split()) <= 4, f"{prompt!r}: over-long {unit!r}"


def test_empty_prompt():
    assert extract_found_prompts("") == []


def test_max_items_respected():
    prompt = ("A monkey from China with a blue face and golden fur, adapted to "
              "cold climates")
    assert len(extract_found_prompts(prompt, max_items=2)) <= 2
