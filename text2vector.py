#!/usr/bin/env python3
# text2vector.py
# Generate vectorized JSON from timestamped transcript .txt files for Solr.
# - Robust timestamp parsing (HH:MM:SS or MM:SS).
# - Sliding-window chunking with overlap.
# - Optional single-file output.

import argparse
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import List, Dict, Any

import torch
from sentence_transformers import SentenceTransformer


TIMESTAMP_RE = re.compile(r'^\s*(\d{1,2}:\d{2})(?::\d{2})?\s*$')


def get_device() -> str:
    """Automatically select the best available device."""
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def is_timestamp(line: str) -> bool:
    """Check if a line is a timestamp in format HH:MM:SS or MM:SS."""
    return bool(TIMESTAMP_RE.match(line))


def clean_text(s: str) -> str:
    """
    Minimal cleanup for Solr: collapse whitespace; strip stray BOMs or controls.
    Avoid aggressive normalization so we keep clinical phrasing intact.
    """
    s = s.replace('\ufeff', ' ').strip()
    s = re.sub(r'\s+', ' ', s)
    return s


def parse_transcript(
    file_path: Path,
    file_name: str,
    model: SentenceTransformer,
    chunk_size: int = 3,
    overlap: int = 0,
) -> List[Dict[str, Any]]:
    """
    Read a transcript and produce chunk records:
      id, file_name, timestamp, sentence (text), vector (list[float])
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"[WARN] Error reading file {file_path}: {e}")
        return []

    # If first lines are metadata (common in exports), allow skipping empties
    # and keep all non-empty lines.
    lines = [ln.rstrip('\n') for ln in lines]

    # 1) Group by timestamps → [{timestamp, sentence}]
    all_sentences: List[Dict[str, str]] = []
    current_timestamp = None
    current_text: List[str] = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if is_timestamp(line):
            # Flush previous block
            if current_timestamp and current_text:
                blk = clean_text(" ".join(current_text))
                if blk:
                    all_sentences.append({"timestamp": current_timestamp, "sentence": blk})
            current_timestamp = line
            current_text = []
        else:
            # Optional: strip speaker labels like "Dr. X:" or "Speaker 1:"
            # line = re.sub(r'^[A-Za-z][\w\s\.-]{0,30}:\s+', '', line)
            current_text.append(line)

    # Flush tail
    if current_timestamp and current_text:
        blk = clean_text(" ".join(current_text))
        if blk:
            all_sentences.append({"timestamp": current_timestamp, "sentence": blk})

    if not all_sentences:
        return []

    # 2) Sliding window chunking
    if chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("--overlap must satisfy 0 <= overlap < chunk_size")

    step = chunk_size - overlap
    chunked: List[Dict[str, Any]] = []

    for i in range(0, len(all_sentences), step):
        chunk = all_sentences[i:i + chunk_size]
        if not chunk:
            continue

        combined_text = clean_text(" ".join(c['sentence'] for c in chunk))
        if not combined_text:
            continue

        start_timestamp = chunk[0]['timestamp']

        # 3) Encode to plain Python floats (avoid numpy types)
        vec = model.encode(combined_text, convert_to_numpy=True).astype(float).tolist()

        chunked.append({
            "id": str(uuid.uuid4()),
            "file_name": file_name,            # match Solr schema
            "timestamp": start_timestamp,      # match Solr schema
            "sentence": combined_text,         # match Solr schema
            "vector": vec,                     # match Solr schema
        })

    return chunked


def main():
    parser = argparse.ArgumentParser(
        description="Process transcript .txt files into vector embeddings for Solr."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing transcript .txt files."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write JSON outputs."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="SentenceTransformer model (HuggingFace id)."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=3,
        help="Number of timestamped sentences per chunk."
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Overlap sentences between consecutive chunks (must be < chunk-size)."
    )
    parser.add_argument(
        "--single-file",
        action="store_true",
        help="Write a single combined output.json instead of per-file JSONs."
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.txt",
        help="Glob to select input files (default: *.txt)."
    )
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Select device and load model once
    device = get_device()
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading model: {args.model_name} ...")
    model = SentenceTransformer(args.model_name, device=device)
    print("[INFO] Model loaded.")

    # Enumerate input files (default: *.txt, skip hidden)
    files = [p for p in sorted(in_dir.glob(args.glob)) if p.is_file() and not p.name.startswith(".")]
    if not files:
        print(f"[INFO] No files matched {in_dir / args.glob}. Nothing to do.")
        sys.exit(0)

    combined: List[Dict[str, Any]] = []

    for i, fp in enumerate(files, 1):
        try:
            print(f"[INFO] Processing {i}/{len(files)}: {fp.name}")
            file_name_base = fp.stem
            records = parse_transcript(
                file_path=fp,
                file_name=file_name_base,
                model=model,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
            )
            if not records:
                print(f"[WARN]   No records produced for {fp.name}.")
                continue

            if args.single_file:
                combined.extend(records)
            else:
                out_path = out_dir / f"{file_name_base}.json"
                with open(out_path, "w", encoding="utf-8") as jf:
                    json.dump(records, jf, indent=2)
        except Exception as e:
            print(f"[ERROR] Failed to process {fp.name}: {e}")

    if args.single_file:
        out_path = out_dir / "output.json"
        with open(out_path, "w", encoding="utf-8") as jf:
            json.dump(combined, jf, indent=2)
        print(f"[INFO] Wrote combined file: {out_path}")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
