"""
Export dialogue chat history as a plain text file.

Standalone: stdlib only, no imports from other SpeechtoText packages.
Importing this module has no side effects. Saving does not mutate the dialogue list.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _normalize_turns(dialogue: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build a clean snapshot; does not modify the original list or dicts."""
    out: list[dict[str, str]] = []
    for item in dialogue:
        if not isinstance(item, dict):
            continue
        sp = item.get("speaker")
        tx = (item.get("text") or "").strip()
        if not tx:
            continue
        out.append({
            "speaker": "" if sp is None else str(sp),
            "text": tx,
        })
    return out


def _default_export_path() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path("chat_exports") / f"chat_{ts}.txt"


def save_chat_history(
    dialogue: list[dict[str, Any]],
    path: str | Path | None = None,
) -> Path:
    """
    Write chat turns as readable plain text. Does not modify `dialogue` or its elements.

    Each turn is written as:
        Speaker: what they said

    Args:
        dialogue: List of {"speaker": str | None, "text": str} (same shape as main's history).
        path: Target .txt file. If None, uses ./chat_exports/chat_<UTC time>.txt

    Returns:
        Resolved path written.

    Raises:
        OSError: Write or mkdir failure.
    """
    turns = _normalize_turns(dialogue)
    target = Path(path) if path else _default_export_path()
    target = target.expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        f"Chat export\n",
        f"Saved: {now}\n",
        f"Turns: {len(turns)}\n",
        f"{'-' * 40}\n\n",
    ]
    for t in turns:
        sp = t["speaker"] or "Speaker"
        lines.append(f"{sp}: {t['text']}\n\n")

    target.write_text("".join(lines), encoding="utf-8")
    return target.resolve()
