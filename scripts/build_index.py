"""Run once to download the corpus (if needed) and build the vector index.

Usage: python scripts/build_index.py [--strategy recursive|fixed|semantic]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from corpus.download import main as download_corpus
from pipeline.indexer import build_index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy", default="recursive", choices=["fixed", "recursive", "semantic"],
        help="Chunking strategy to use for the main index (default: recursive)",
    )
    args = parser.parse_args()

    print("=== Step 1: Download corpus ===")
    download_status = download_corpus()
    if download_status != 0:
        print("No corpus PDFs available - aborting index build.")
        return 1

    print(f"\n=== Step 2: Build index (strategy={args.strategy}) ===")
    store = build_index(strategy=args.strategy)
    stats = store.get_stats()
    print(f"\nIndex build complete: {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
