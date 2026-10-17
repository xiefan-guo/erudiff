"""Extraction of y_found (foundational sub-prompts) from implicit prompts.

y_found serves anti-forgetting knowledge consolidation (AF-KC): the
DK-DM(y_found, y_found) term teaches nothing new, it pins the model's existing
rendering of the bare entities ("Albert Einstein", "musical instruments") so
that refactoring a composite fact does not erode its parts. The extractor
therefore only has to be conservative and on-topic, and precision matters much
more than recall.

The paper annotates y_found with a language model under this policy:

  1. Keep wording from the original prompt whenever possible.
  2. Prefer short noun units ("A bird", "The shield", "Captain America").
  3. If several valid entities exist, include several items.
  4. If uncertain, prefer dropping ambiguous items.

  Never include abstract nouns ("invention", "moment", "symbol", "concept"),
  verbs, clauses, full sentences, or bare adjectives and adverbs.

  Example: "The shield used by Captain America" -> ["The shield", "Captain America"]

`extract_found_prompts` implements the same policy with rules so that data
preparation runs offline; `extract_found_prompts_llm` is the extension point for
an API-backed extractor.
"""

from __future__ import annotations

import re

# Determiners that may lead a short noun unit (rule 2 keeps them).
_LEADING_DETERMINERS = {"a", "an", "the"}

# Abstract nouns the policy forbids, plus the generic scene vocabulary that
# knowledge benchmarks pad prompts with.
_ABSTRACT_NOUNS = {
    "invention", "moment", "symbol", "concept", "idea", "notion", "theme",
    "state", "states", "process", "phenomenon", "phenomena", "behavior",
    "behaviour", "characteristic", "characteristics", "feature", "features",
    "aspect", "example", "instance", "kind", "type", "sort", "way", "form",
    "style", "design", "structure", "construction", "condition", "situation",
    "setting", "scene", "view", "image", "picture", "photo", "photograph",
    "depiction", "illustration", "representation", "rendering", "background",
    "foreground", "detail", "details", "quality", "heritage", "artistry",
    "culture", "commonsense", "knowledge", "reasoning", "aspiration",
    "aspirations", "tradition", "custom", "customs", "event", "events",
    "activity", "activities", "celebration", "time", "day", "night", "year",
    "years", "decade", "decades", "century", "degree", "degrees", "amount",
    "number", "level", "point", "part", "side", "top", "bottom", "left",
    "right", "middle", "center", "centre", "one", "thing", "things", "place",
    "places", "area", "region", "environment", "surroundings",
}

# Clause and modifier markers, including pronouns left dangling when a
# possessive or relative clause is split.
_CLAUSE_MARKERS = {
    "is", "are", "was", "were", "be", "being", "been", "has", "have", "had",
    "do", "does", "did", "will", "would", "can", "could", "should", "may",
    "might", "must", "that", "which", "who", "whom", "whose", "where", "when",
    "while", "because", "if", "than", "then", "as", "used", "shown", "showing",
    "depict", "depicting", "depicted", "highlight", "highlighting", "generate",
    "show", "showcasing", "exhibiting", "emphasizing", "featuring", "known",
    "referred", "often", "most", "more", "very", "also", "just", "only",
    "it", "its", "they", "them", "their", "he", "him", "his", "she", "her",
    "we", "us", "our", "you", "your", "i", "my",
}

# Style and adjective padding that commonly trails these prompts.
_STYLE_WORDS = {
    "realistic", "realism", "simple", "clear", "clean", "plain", "high",
    "photorealistic", "detailed", "iconic", "famous", "traditional", "typical",
    "common", "distinctive", "modern", "ancient", "national", "raw",
    "undeveloped", "unripe", "empty", "completely", "favorite", "favourite",
    "popular", "best", "worst", "first", "last", "main", "primary", "key",
    "specific", "certain", "various", "diverse", "complex",
}

# Verbs that, as a span's head, mean a predicate was captured rather than an
# entity ("A structure that touches the sky" -> "A touches the sky").
_VERB_HEADS = {
    "touches", "touch", "holds", "hold", "holding", "sits", "sit", "sitting",
    "stands", "stand", "standing", "filled", "fill", "left", "kept", "keep",
    "watered", "exposed", "adapted", "lifting", "pulled", "played", "shared",
    "grapple", "surfaces", "looking", "coming", "says", "make", "makes",
}

_STOPWORDS = _CLAUSE_MARKERS | _STYLE_WORDS

# Keeps hyphens and apostrophes inside words ("Snub-nosed", "Einstein's").
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-']*")

_MAX_UNIT_WORDS = 4


def _strip_possessive(word: str) -> str:
    """"Einstein's" -> "Einstein"; "leaves'" -> "leaves"."""
    if word.lower().endswith("'s"):
        return word[:-2]
    if word.endswith("'"):
        return word[:-1]
    return word


def _normalise_unit(words: list[str]) -> str | None:
    """Trim a candidate word span into a keepable short noun unit, or None."""
    # A conjunction joins two separate entities ("face and golden fur"); keep the
    # first and let the caller mine the rest from its own chunking.
    for idx, word in enumerate(words):
        if word.lower() in {"and", "or"}:
            words = words[:idx]
            break

    # Stopwords are dropped anywhere in the span, since style padding also shows
    # up mid-unit ("a simple space" -> "a space").
    words = [w for w in words if w.lower() not in _STOPWORDS]
    while words and words[-1].lower() in _LEADING_DETERMINERS:
        words = words[:-1]
    if not words:
        return None

    if len(words) > _MAX_UNIT_WORDS:
        lead = [words[0]] if words[0].lower() in _LEADING_DETERMINERS else []
        words = lead + words[-(_MAX_UNIT_WORDS - len(lead)):]

    words = [_strip_possessive(w) for w in words]
    words = [w for w in words if w]
    if not words:
        return None

    # The head noun carries the unit's meaning; reject abstract and verbal heads.
    head = words[-1].lower()
    if head in _ABSTRACT_NOUNS or head in _STOPWORDS or head in _VERB_HEADS:
        return None
    if any(w.lower() in _VERB_HEADS for w in words):
        return None
    if head in _LEADING_DETERMINERS:
        return None
    if len(words) == 1 and len(head) < 3:
        return None

    unit = " ".join(words)
    if unit.lower() in _LEADING_DETERMINERS:
        return None
    return unit


def extract_found_prompts(prompt: str, max_items: int = 3) -> list[str]:
    """Extract short entity / proper-noun units from an implicit prompt.

    Returns at most `max_items` de-duplicated units in order of appearance, or an
    empty list when nothing survives the policy (AF-KC skips the y_found term for
    such samples).
    """
    if not prompt:
        return []

    units: list[str] = []
    seen: set[str] = set()

    def _content_key(unit: str) -> tuple[str, ...]:
        """Unit identity ignoring determiners, so "the stone" == "stone"."""
        return tuple(w.lower() for w in unit.split()
                     if w.lower() not in _LEADING_DETERMINERS)

    def push(unit: str | None) -> None:
        if not unit:
            return
        key = unit.lower()
        if key in seen:
            return
        # Suppress a unit that is a sub-span of one already kept, or that
        # subsumes one; pass 1 runs first and holds the higher-confidence items.
        new_key = _content_key(unit)
        if not new_key:
            return
        for kept in units:
            kept_key = _content_key(kept)
            if new_key == kept_key:
                return
            if set(new_key) <= set(kept_key) or set(kept_key) <= set(new_key):
                return
        seen.add(key)
        units.append(unit)

    # Pass 1: proper-noun runs ("Captain America", "Mid-Autumn Festival"). A
    # single capitalised word in sentence-initial position is just sentence case.
    tokens = _TOKEN_RE.findall(prompt)
    i = 0
    while i < len(tokens):
        if tokens[i][:1].isupper() and tokens[i].lower() not in _STOPWORDS:
            j = i
            run: list[str] = []
            while (j < len(tokens) and tokens[j][:1].isupper()
                   and tokens[j].lower() not in _STOPWORDS):
                run.append(tokens[j])
                j += 1
            if not (i == 0 and len(run) == 1):
                push(_normalise_unit(run))
            i = j
        else:
            i += 1

    # Pass 2: determiner-led noun units. Splitting on punctuation, prepositions,
    # conjunctions, participles and relative pronouns leaves at most one head
    # noun per chunk. A possessive also ends a chunk, which is what splits
    # "Einstein's favorite musical instruments" into owner and owned.
    chunks = re.split(
        r"[,;:.]"
        r"|'s\b"
        r"|\b(?:of|in|on|at|with|without|from|for|to|by|near|inside|beside"
        r"|during|under|over|into|and|or"
        r"|exhibiting|showing|showcasing|highlighting|emphasizing|featuring"
        r"|depicting|holding|containing|that|which|who|when|while)\b",
        prompt,
        flags=re.IGNORECASE,
    )
    for chunk in chunks:
        words = _TOKEN_RE.findall(chunk)
        if not words:
            continue
        if words[0].lower() in _LEADING_DETERMINERS or len(words) <= _MAX_UNIT_WORDS:
            push(_normalise_unit(list(words)))

    return units[:max_items]


def extract_found_prompts_llm(prompt: str) -> list[str]:  # pragma: no cover
    """Extension point for model-based y_found extraction.

    Implement this with the policy in the module docstring to reproduce the
    paper's annotation pipeline, then pass `--found-extractor llm` to
    `tools/prepare_data.py`.
    """
    raise NotImplementedError(
        "Model-based extraction is not configured; use the rule-based "
        "extract_found_prompts or implement this hook against your own API."
    )
