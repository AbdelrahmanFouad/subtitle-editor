#!/usr/bin/env python3
import re
import sys
import argparse
from pathlib import Path
from textwrap import shorten

# Regex to match SRT timestamp lines
TIMESTAMP_RE = re.compile(
    r'^(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2},\d{3})$'
)
# Regex to detect any Arabic character
ARABIC_RE = re.compile(r'[\u0600-\u06FF]')

def make_key(name: str) -> str:
    """Match files by their first 15 characters."""
    return name[:15]

def normalize_text(lines: list[str]) -> str:
    """
    Strip each line, collapse internal whitespace, join with a single space.
    Only called on lines that have already been filtered to exclude Arabic.
    """
    return " ".join(" ".join(l.split()) for l in lines).strip()

def parse_srt(path: Path) -> list[dict]:
    """
    Parse an SRT‐style file into a list of entries:
    [{'index': int, 'timestamp': str, 'text_lines': [str,...], 'norm_text': str}, ...]
    Uses utf-8-sig to strip BOM if present.
    """
    raw = path.read_text(encoding='utf-8-sig').splitlines()
    entries = []
    buf = []
    for line in raw + [""]:
        if line.strip() == "" and buf:
            # buf[0] = index, buf[1] = timestamp, buf[2:] = text lines
            idx_line = buf[0].lstrip('\ufeff')
            try:
                idx = int(idx_line)
            except ValueError:
                sys.stderr.write(f"Warning: invalid index '{buf[0]}' in {path.name}, skipping block\n")
                buf = []
                continue

            ts = buf[1]
            text_lines = buf[2:]

            # Filter out Arabic lines for matching
            english_lines = [l for l in text_lines if not ARABIC_RE.search(l)]
            norm = normalize_text(english_lines)

            entries.append({
                "index": idx,
                "timestamp": ts,
                "text_lines": text_lines,
                "norm_text": norm
            })
            buf = []
        else:
            buf.append(line)
    return entries

def write_srt(entries: list[dict], out_path: Path):
    """Write entries back to SRT format."""
    with out_path.open('w', encoding='utf-8') as f:
        for e in entries:
            f.write(f"{e['index']}\n")
            f.write(f"{e['timestamp']}\n")
            for l in e['text_lines']:
                f.write(f"{l}\n")
            f.write("\n")

def main():
    p = argparse.ArgumentParser(
        description="Copy timestamps from new→old by matching only English text lines."
    )
    p.add_argument('new_dir', type=Path, help="Dir with correct‑timestamp SRTs")
    p.add_argument('old_dir', type=Path, help="Dir with translated SRTs")
    p.add_argument('--out-dir', type=Path, default=None,
                   help="If set, write updated files here; otherwise overwrite old files")
    args = p.parse_args()

    # validate dirs
    if not args.new_dir.is_dir() or not args.old_dir.is_dir():
        p.error("Both new_dir and old_dir must be existing directories.")
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    # map new files by 15‑char key
    new_map: dict[str, Path] = {}
    for nf in args.new_dir.iterdir():
        if nf.is_file():
            k = make_key(nf.name)
            if k in new_map:
                sys.stderr.write(f"Warning: duplicate key {k!r}, using {nf.name}\n")
            new_map[k] = nf

    # process old files
    for of in args.old_dir.iterdir():
        if not of.is_file():
            continue
        key = make_key(of.name)
        nf  = new_map.get(key)
        if not nf:
            sys.stderr.write(f"Skipping {of.name}: no new file for key '{key}'\n")
            continue

        new_entries = parse_srt(nf)
        old_entries = parse_srt(of)

        # build text→list[timestamp]
        ts_map: dict[str, list[str]] = {}
        for e in new_entries:
            # Only entries with non-empty English text will be in ts_map
            if e['norm_text']:
                ts_map.setdefault(e['norm_text'], []).append(e['timestamp'])

        # walk old entries, replacing where we can
        for e in old_entries:
            txt = e['norm_text']
            if not txt or txt not in ts_map or not ts_map[txt]:
                print(f"{of.name}: entry {e['index']} has no match for English text "
                      f"\"{shorten(txt, width=30)}\"")
                continue
            lst = ts_map[txt]
            if len(lst) > 1:
                sys.stderr.write(
                    f"Warning: {of.name} entry {e['index']} text is duplicated in new file; "
                    "using next timestamp\n"
                )
            e['timestamp'] = lst.pop(0)

        # write out
        out_path = (args.out_dir / of.name) if args.out_dir else of
        write_srt(old_entries, out_path)
        print(f"{of.name}: written updated timestamps to {out_path}")

if __name__ == '__main__':
    main()
