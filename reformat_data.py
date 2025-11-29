#!/usr/bin/env python3
"""Simple reformatter for data.txt:
- creates a backup `data.txt.bak`
- collapses excessive line breaks and joins broken words by
  collapsing all consecutive whitespace into single spaces,
 while preserving paragraph breaks (multiple blank lines).
"""
import re
from pathlib import Path


def main():
    root = Path(__file__).parent
    data_path = root / "data.txt"
    backup_path = root / "data.txt.bak"

    if not data_path.exists():
        print(f"File not found: {data_path}")
        return

    raw = data_path.read_text(encoding="utf-8")

    # normalize newlines
    raw = raw.replace('\r\n', '\n')

    # mark paragraph breaks: two or more newlines -> placeholder
    marked = re.sub(r"\n\s*\n+", "<PARA>", raw)

    # collapse all remaining whitespace to single spaces (joins broken words)
    collapsed = re.sub(r"\s+", " ", marked)

    # restore paragraph breaks
    restored = collapsed.replace("<PARA>", "\n\n")

    cleaned = restored.strip() + "\n"

    # backup original
    backup_path.write_text(raw, encoding="utf-8")

    # write cleaned version
    data_path.write_text(cleaned, encoding="utf-8")

    print(f"Backed up original to: {backup_path}")
    print(f"Wrote cleaned file to: {data_path}")


if __name__ == "__main__":
    main()
