"""Parse single or multiple words from text, files, or stdin."""

from __future__ import annotations

from pathlib import Path


def parse_word_list(text: str) -> list[str]:
    """Parse words separated by newlines and/or commas."""
    if not text or not text.strip():
        return []

    if "\n" in text:
        items: list[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "," in line:
                items.extend(part.strip() for part in line.split(",") if part.strip())
            else:
                items.append(line)
        return items

    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]

    return [text.strip()]


def parse_word_list_from_file(path: str | Path) -> list[str]:
    content = Path(path).read_text(encoding="utf-8")
    return parse_word_list(content)
