#!/usr/bin/env python3
"""
Export a trained sklearn DecisionTreeRegressor to a plain-text file
readable by the C++ TreeEvaluator.

Text format:
  Line 1: <n_nodes> <n_features>
  Lines 2..(n_nodes+1): <feature> <threshold> <left_child> <right_child> <value>

Where:
  feature == -2  →  leaf node (sklearn TREE_LEAF constant)
  value          →  predicted win probability (regression leaf value)

Usage:
    python scripts/export_tree_cpp.py --depth 5 [--model-dir data] [--output-dir data]
    python scripts/export_tree_cpp.py --all [--model-dir data] [--output-dir data]
"""

import argparse
import os

import joblib
import numpy as np


def export_tree(joblib_path: str, txt_path: str) -> None:
    bundle = joblib.load(joblib_path)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    tree = model.tree_

    n_nodes = tree.node_count
    n_features = len(feature_cols)

    # sklearn TREE_LEAF = -1, TREE_UNDEFINED = -2.
    # feature[leaf] == TREE_UNDEFINED (-2); we use -2 as the leaf sentinel.
    features   = tree.feature          # shape (n_nodes,)
    thresholds = tree.threshold        # shape (n_nodes,)
    left       = tree.children_left    # shape (n_nodes,)
    right      = tree.children_right   # shape (n_nodes,)
    # tree.value has shape (n_nodes, n_outputs, max_n_classes).
    # For a single-output regressor this is (n_nodes, 1, 1).
    values = tree.value[:, 0, 0]      # shape (n_nodes,)

    with open(txt_path, "w") as f:
        f.write(f"{n_nodes} {n_features}\n")
        for i in range(n_nodes):
            f.write(f"{features[i]} {thresholds[i]:.8f} {left[i]} {right[i]} {values[i]:.8f}\n")

    print(f"  Exported {n_nodes} nodes, {n_features} features → {txt_path}")
    print(f"  Features: {feature_cols}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export sklearn decision tree to C++-readable text format"
    )
    parser.add_argument("--depth", type=int, default=None,
                        help="Export model for a specific depth")
    parser.add_argument("--all", action="store_true",
                        help="Export all tree_model_d*.joblib files found")
    parser.add_argument("--model-dir", default="data",
                        help="Directory containing .joblib model files")
    parser.add_argument("--output-dir", default="data",
                        help="Directory to write .txt model files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.all:
        import glob
        paths = sorted(glob.glob(os.path.join(args.model_dir, "tree_model_d*.joblib")))
        if not paths:
            print(f"No tree_model_d*.joblib files found in {args.model_dir}")
            return
        for p in paths:
            depth = os.path.basename(p).removeprefix("tree_model_d").removesuffix(".joblib")
            txt = os.path.join(args.output_dir, f"tree_model_d{depth}.txt")
            print(f"Exporting depth={depth}:")
            export_tree(p, txt)
    elif args.depth is not None:
        joblib_path = os.path.join(args.model_dir, f"tree_model_d{args.depth}.joblib")
        txt_path    = os.path.join(args.output_dir, f"tree_model_d{args.depth}.txt")
        print(f"Exporting depth={args.depth}:")
        export_tree(joblib_path, txt_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
