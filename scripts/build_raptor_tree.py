"""Build a RAPTOR hierarchical summary tree over an already-indexed corpus.

Run scripts/build_index.py first. Each internal tree node costs one Gemini
summarization call, paced by the same rate limiter as core/generation.py -
expect this to take a while on the free tier; it's a one-time offline step,
not something run on the query path.

Usage: python scripts/build_raptor_tree.py [--cluster-size 5] [--max-levels 3]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.embeddings import EmbeddingModel
from core.generation import RAGGenerator
from core.raptor import DEFAULT_CLUSTER_SIZE, DEFAULT_MAX_LEVELS, build_raptor_tree, save_tree
from core.vectorstore import ChromaStore

DEFAULT_TREE_PATH = Path(__file__).parent.parent / "vectorstore_data" / "raptor_tree.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster-size", type=int, default=DEFAULT_CLUSTER_SIZE)
    parser.add_argument("--max-levels", type=int, default=DEFAULT_MAX_LEVELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_TREE_PATH)
    args = parser.parse_args()

    store = ChromaStore()
    documents = store.get_all_documents()
    if not documents:
        print("No chunks in the vector store - run scripts/build_index.py first.")
        return 1
    print(f"Building RAPTOR tree over {len(documents)} leaf chunks "
          f"(cluster_size={args.cluster_size}, max_levels={args.max_levels})...")

    nodes = build_raptor_tree(
        documents, store.embedder, RAGGenerator(),
        cluster_size=args.cluster_size, max_levels=args.max_levels,
    )
    save_tree(nodes, args.output)

    by_level: dict[int, int] = {}
    for n in nodes:
        by_level[n.level] = by_level.get(n.level, 0) + 1
    print(f"Tree saved to {args.output}. Nodes per level: {by_level}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
