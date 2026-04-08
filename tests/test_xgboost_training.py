from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from xgboost_training import (
    CUSTOMER_INDUSTRY,
    build_features,
    compute_days_offset_from_due_date,
    customer_to_industry,
    load_invoices,
    make_preprocessor,
    run_training,
    train_val_test_split,
)


@pytest.mark.parametrize(
    ("paid", "due", "expected"),
    [
        ("2026-02-15", "2026-02-15", 0.0),
        ("2026-02-20", "2026-02-15", 5.0),
        ("2026-02-10", "2026-02-15", -5.0),
    ],
)
def test_compute_days_offset_expected_values(
    paid: str,
    due: str,
    expected: float,
) -> None:
    y = compute_days_offset_from_due_date(pd.Series([paid]), pd.Series([due]))
    assert y.dtype.kind == "f"
    assert y[0] == expected


def test_compute_days_offset_invalid_raises() -> None:
    paid = pd.Series(["2026-02-10", None])
    due = pd.Series(["2026-02-15", "2026-02-15"])
    with pytest.raises(ValueError, match="paid_date or due_date"):
        compute_days_offset_from_due_date(paid, due)


def test_customer_to_industry_known(sample_invoices: pd.DataFrame) -> None:
    ind = customer_to_industry(sample_invoices.loc[[0], "customer"])
    assert ind.iloc[0] == CUSTOMER_INDUSTRY["James Patel"]


def test_customer_to_industry_unknown() -> None:
    ind = customer_to_industry(pd.Series(["Not A Real Customer"]))
    assert ind.iloc[0] == "unknown"


def test_build_features_drops_paid_and_customer(
    sample_invoices: pd.DataFrame,
) -> None:
    X, y = build_features(sample_invoices)
    assert "paid_date" not in X.columns
    assert "customer" not in X.columns
    assert "invoice_id" not in X.columns
    assert "creation_date" not in X.columns
    assert "due_date" not in X.columns
    assert "industry" in X.columns
    expected_y = np.array([2.0, 4.0])
    np.testing.assert_array_almost_equal(y, expected_y)


def test_build_features_invoice_num_from_id(sample_invoices: pd.DataFrame) -> None:
    X, _ = build_features(sample_invoices)
    assert X["invoice_num"].tolist() == [1001.0, 1002.0]


def test_train_val_test_split_proportions() -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "a": np.arange(100, dtype=float),
            "b": rng.normal(size=100),
        }
    )
    y = np.arange(100, dtype=float)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y, random_state=42
    )
    assert len(X_train) == 70
    assert len(X_val) == 10
    assert len(X_test) == 20
    assert len(y_train) == 70 and len(y_val) == 10 and len(y_test) == 20
    # check partitioning (no overlap, no row loss)
    train_idx = set(X_train.index.tolist())
    val_idx = set(X_val.index.tolist())
    test_idx = set(X_test.index.tolist())
    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)
    assert len(train_idx | val_idx | test_idx) == len(X)


def test_train_val_test_split_is_deterministic() -> None:
    X = pd.DataFrame({"a": np.arange(50)})
    y = np.arange(50, dtype=float)
    s1 = train_val_test_split(X, y, random_state=7)
    s2 = train_val_test_split(X, y, random_state=7)
    np.testing.assert_array_equal(s1[0].index.to_numpy(), s2[0].index.to_numpy())
    np.testing.assert_array_equal(s1[1].index.to_numpy(), s2[1].index.to_numpy())
    np.testing.assert_array_equal(s1[2].index.to_numpy(), s2[2].index.to_numpy())


def test_load_invoices_rejects_missing(tmp_path: Path) -> None:
    bad = pd.DataFrame({"x": [1.0, np.nan]})
    p = tmp_path / "bad.csv"
    bad.to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing"):
        load_invoices(p)


def test_make_preprocessor_fit_transform(sample_invoices: pd.DataFrame) -> None:
    X, _ = build_features(sample_invoices)
    prep = make_preprocessor(X)
    out = prep.fit_transform(X)
    assert out.shape[0] == len(X)
    assert not np.any(np.isnan(out))
    # expected dimensions: 6 numeric + one-hot columns for 4 categorical features
    names = prep.get_feature_names_out()
    assert len(names) == out.shape[1]
    assert any(name.startswith("cat__payment_method_") for name in names)
    assert any(name.startswith("cat__payment_terms_") for name in names)
    assert any(name.startswith("cat__company_") for name in names)
    assert any(name.startswith("cat__industry_") for name in names)


def test_metrics_dict_keys_if_train_runs_quick(tmp_path: Path, sample_invoices: pd.DataFrame) -> None:
    pytest.importorskip("xgboost")
    csv = tmp_path / "tiny.csv"
    # enough rows for cv n_splits >= 2 on train (~70% of 20 >= 14, need n_train>=3)
    big = pd.concat([sample_invoices] * 10, ignore_index=True)
    csv.write_text(big.to_csv(index=False))
    model_path = tmp_path / "m.joblib"
    _, metrics = run_training(
        csv_path=csv,
        model_path=model_path,
        n_iter=1,
        random_state=42,
    )
    assert metrics["target"] == "days_offset_from_due_date"
    assert "val_mae" in metrics
    assert "test_r2" in metrics
    assert model_path.is_file()
    assert Path(metrics["model_path"]).resolve() == model_path.resolve()
    assert np.isfinite(metrics["val_mae"])
    assert np.isfinite(metrics["test_mae"])


def test_run_training_raises_on_too_few_rows(tmp_path: Path, sample_invoices: pd.DataFrame) -> None:
    pytest.importorskip("xgboost")
    csv = tmp_path / "tiny.csv"
    # 2 rows triggers an early split-size failure (or later CV-size failure).
    sample_invoices.to_csv(csv, index=False)
    with pytest.raises(
        ValueError,
        match="Not enough training rows|resulting train set will be empty",
    ):
        run_training(csv_path=csv, n_iter=1, random_state=42)
