"""Dataset of (implicit, explicit, foundational) prompt triplets.

Each iteration samples a triplet (y_impl, y_expl, y_found) and the AF-KC mixer
decides which pair is trained on (Algorithm 1, line 4).

Expected JSONL fields, as produced by `tools/prepare_data.py`:
    id          unique sample id
    prompt      y_impl  -- implicit prompt requiring world knowledge
    aug_prompt  y_expl  -- explicit anchor naming the answer directly
    sub_prompt  list of y_found candidates (entity / proper nouns)
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from torch.utils.data import Dataset


class PromptTripletDataset(Dataset):
    """Serves (y_impl, y_expl, y_found) triplets.

    One y_found is drawn per access from the sample's candidate list, so across
    epochs a sample exercises all of its foundational anchors instead of binding
    permanently to the first one. Samples with no candidates yield an empty
    string, and `PromptMixer` falls back to the explicit consolidation term.
    """

    def __init__(self, jsonl_path: str | Path, seed: int = 0):
        self.path = Path(jsonl_path)
        if not self.path.exists():
            raise FileNotFoundError(f"training data not found: {self.path}")

        self.records: list[dict] = []
        with open(self.path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{self.path}:{lineno}: bad JSON: {exc}") from exc

                missing = {"prompt", "aug_prompt"} - item.keys()
                if missing:
                    raise ValueError(
                        f"{self.path}:{lineno}: missing field(s) {sorted(missing)}"
                    )

                self.records.append({
                    "id": str(item.get("id", lineno)),
                    "impl": item["prompt"],
                    "expl": item["aug_prompt"],
                    "found": list(item.get("sub_prompt") or []),
                })

        if not self.records:
            raise ValueError(f"no usable records in {self.path}")

        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        rec = self.records[index]
        found = self._rng.choice(rec["found"]) if rec["found"] else ""
        return {
            "id": rec["id"],
            "impl": rec["impl"],
            "expl": rec["expl"],
            "found": found,
        }

    @property
    def coverage(self) -> dict[str, int]:
        """How many samples carry at least one y_found candidate."""
        with_found = sum(1 for r in self.records if r["found"])
        return {
            "total": len(self.records),
            "with_found": with_found,
            "without_found": len(self.records) - with_found,
        }
