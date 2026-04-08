#!/usr/bin/env python3
"""
Permutation importance for the saved ``days_offset_from_due_date`` pipeline.

Shuffles each raw column in the feature matrix and measures how much
``neg_mean_absolute_error`` drops (higher reported mean = more important column).

Uses the same ``build_features`` / ``train_val_test_split`` as training
(``random_state=42`` by default) and evaluates on the **validation** split.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from xgboost_training import (  # noqa: E402
    build_features,
    load_invoices,
    train_val_test_split,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models" / "xgb_days_offset.joblib",
        help="Fitted sklearn Pipeline (joblib)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "data" / "invoices.csv",
        help="Invoice CSV",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=10,
        help="Permutation repeats per feature",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Split and permutation RNG seed (match training for same val rows)",
    )
    args = parser.parse_args()

    if not args.model.is_file():
        raise SystemExit(f"Model not found: {args.model}")

    model = joblib.load(args.model)
    df = load_invoices(args.csv)
    X, y = build_features(df)
    _, X_val, _, _, y_val, _ = train_val_test_split(
        X, y, random_state=args.random_state
    )

    result = permutation_importance(
        model,
        X_val,
        y_val,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
        n_jobs=-1,
        scoring="neg_mean_absolute_error",
    )

    table = pd.DataFrame(
        {
            "feature": X_val.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    print("Target: days_offset_from_due_date (paid_date - due_date, days)")
    print(
        "Metric: permutation importance with scoring=neg_mean_absolute_error\n"
        "        (larger mean = shuffling this column hurts MAE more)\n"
    )
    print(table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(
        "\nNote: small or negative means can appear by chance (finite val set / repeats);"
        " increase --n-repeats for stability."
    )


if __name__ == "__main__":
    main()
