"""Data cleaning pipeline nodes.

Each node processes a list of records (dicts) and returns filtered/cleaned records.
Nodes are composable and run sequentially in a pipeline.
"""

import hashlib
import re
from collections.abc import Callable

# ═══════════════════════════════════════════════
# Pipeline Runner
# ═══════════════════════════════════════════════


def run_pipeline(records: list[dict], nodes: list[dict]) -> list[dict]:
    """Run a list of cleaning nodes sequentially on the records."""
    result = list(records)
    for node_config in nodes:
        node_type = node_config.get("node_type", "")
        params = node_config.get("params", {})
        handler = NODE_REGISTRY.get(node_type)
        if handler is None:
            raise ValueError(f"Unknown cleaning node type: {node_type}")
        result = handler(result, **params)
    return result


# ═══════════════════════════════════════════════
# Individual Nodes
# ═══════════════════════════════════════════════


def dedup_node(records: list[dict], *, key: str | None = None, **_kwargs) -> list[dict]:
    """Remove exact-duplicate records.

    If `key` is specified, dedup by that field only.
    Otherwise, dedup by full row hash.
    """
    seen: set[str] = set()
    unique: list[dict] = []
    for rec in records:
        if key and key in rec:
            fingerprint = str(rec[key])
        else:
            fingerprint = hashlib.md5(
                str(sorted(rec.items())).encode(), usedforsecurity=False
            ).hexdigest()
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique.append(rec)
    return unique


def length_filter_node(
    records: list[dict],
    *,
    field: str = "text",
    min_length: int = 0,
    max_length: int = 100_000,
    **_kwargs,
) -> list[dict]:
    """Filter rows by character length of a text field."""
    return [
        rec for rec in records if field in rec and min_length <= len(str(rec[field])) <= max_length
    ]


def regex_filter_node(
    records: list[dict],
    *,
    field: str = "text",
    pattern: str,
    mode: str = "include",
    **_kwargs,
) -> list[dict]:
    """Include or exclude rows matching a regex pattern.

    Args:
        mode: 'include' keeps matching rows; 'exclude' removes them.
    """
    compiled = re.compile(pattern, re.IGNORECASE)
    if mode == "include":
        return [rec for rec in records if field in rec and compiled.search(str(rec[field]))]
    else:
        return [rec for rec in records if field not in rec or not compiled.search(str(rec[field]))]


# Common PII patterns
_PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("PHONE", re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b")),
    ("SSN", re.compile(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b")),
    ("IP_ADDRESS", re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d{4}[-.\s]?){3}\d{4}\b")),
]


def pii_redact_node(
    records: list[dict],
    *,
    fields: list[str] | None = None,
    replacement: str = "[REDACTED]",
    **_kwargs,
) -> list[dict]:
    """Redact PII patterns (email, phone, SSN, IP, credit card) from text fields."""
    target_fields = fields or ["text", "content", "input", "output", "instruction", "response"]
    result: list[dict] = []
    for rec in records:
        new_rec = dict(rec)
        for f in target_fields:
            if f in new_rec and isinstance(new_rec[f], str):
                val = new_rec[f]
                for _name, pat in _PII_PATTERNS:
                    val = pat.sub(replacement, val)
                new_rec[f] = val
        result.append(new_rec)
    return result


def language_filter_node(
    records: list[dict],
    *,
    field: str = "text",
    languages: list[str] | None = None,
    **_kwargs,
) -> list[dict]:
    """Filter rows by detected language.

    Uses a simple heuristic (ASCII ratio) as a lightweight fallback.
    For production, integrate fastText or langdetect.
    """
    if not languages:
        return records

    target_langs = {lang.lower() for lang in languages}

    # Simple heuristic: if 'en' is requested, keep rows with high ASCII ratio
    if "en" in target_langs:
        return [rec for rec in records if field in rec and _ascii_ratio(str(rec[field])) > 0.85]

    # For other languages, pass through (full implementation needs fastText)
    return records


def _ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / len(text)


# ═══════════════════════════════════════════════
# Node Registry
# ═══════════════════════════════════════════════

NODE_REGISTRY: dict[str, Callable] = {
    "dedup": dedup_node,
    "length_filter": length_filter_node,
    "regex_filter": regex_filter_node,
    "pii_redact": pii_redact_node,
    "language_filter": language_filter_node,
}
