"""
Train an XGBoost regressor to predict **days_offset_from_due_date** from non-leaking
invoice features.

``days_offset_from_due_date`` = ``paid_date - due_date`` in whole calendar days
(date-only, midnight-normalized). Positive means paid after due; negative means paid before due.
``paid_date`` is never a model input.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


# Synthetic B2B labels: map each known customer name to an industry (feature engineering).
CUSTOMER_INDUSTRY: dict[str, str] = {
    "Viktor Novak": "Manufacturing",
    "Chen Okafor": "Construction",
    "Liam Murphy": "Property management",
    "James Patel": "Wholesale trade",
    "Sofia Rossi": "Retail - hardware",
    "Yuki Tanaka": "Facilities maintenance",
    "Diego Kowalski": "Landscaping",
    "Marcus Bishop": "Transportation / fleet",
    "Aisha Hernandez": "Hospitality",
    "Olivia Silva": "Real estate development",
    "Zara Al-Farsi": "Education facilities",
    "Ethan Park": "Agriculture",
    "Priya Andersson": "Energy / utilities",
    "Maria Nguyen": "Professional services",
    "Nina Costa": "Municipal / public works",
    "Noah Kim": "Healthcare facilities",
}


def customer_to_industry(customer: pd.Series) -> pd.Series:
    return customer.map(CUSTOMER_INDUSTRY).fillna("unknown").astype(str)


def load_invoices(csv_path: Path | None = None) -> pd.DataFrame:
    path = csv_path or _project_root() / "data" / "invoices.csv"
    df = pd.read_csv(path)
    if df.isna().any().any():
        raise ValueError("Input contains missing values; clean or impute before training.")
    return df


def compute_days_offset_from_due_date(
    paid: pd.Series,
    due: pd.Series,
) -> np.ndarray:
    """
    Target: ``days_offset_from_due_date`` = ``paid_date - due_date`` (calendar days).

    paid_date and due_date are normalized to midnight so the delta is an integer day count.
    """
    paid_dt = pd.to_datetime(paid, errors="coerce")
    due_dt = pd.to_datetime(due, errors="coerce")
    if paid_dt.isna().any() or due_dt.isna().any():
        raise ValueError("paid_date or due_date has invalid or missing values.")
    return (
        (paid_dt.dt.normalize() - due_dt.dt.normalize()).dt.days.astype(float).to_numpy()
    )


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Return ``X`` and ``y`` where ``y`` is **days_offset_from_due_date** only (see module
    docstring). ``paid_date`` is dropped before features are finalized.
    """
    days_offset_from_due_date = compute_days_offset_from_due_date(
        df["paid_date"], df["due_date"]
    )
    feature_df = df.drop(columns=["paid_date"]).copy()
    feature_df["creation_date"] = pd.to_datetime(feature_df["creation_date"])
    feature_df["due_date"] = pd.to_datetime(feature_df["due_date"])
    feature_df["creation_epoch"] = feature_df["creation_date"].astype("int64") // 10**9
    feature_df["due_epoch"] = feature_df["due_date"].astype("int64") // 10**9
    feature_df["invoice_num"] = feature_df["invoice_id"].astype(str).str.extract(
        r"(\d+)", expand=False
    ).astype(float)
    feature_df["industry"] = customer_to_industry(feature_df["customer"])
    feature_df = feature_df.drop(
        columns=[
            "creation_date",
            "due_date",
            "customer",
            "item_description",
            "invoice_id",
        ]
    )
    return feature_df, days_offset_from_due_date


def train_val_test_split(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """70% train, 20% test, 10% validation (of full data)."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=random_state,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=2 / 3,
        random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = [
        "unit_price",
        "quantity",
        "invoice_amount",
        "creation_epoch",
        "due_epoch",
        "invoice_num",
    ]
    categorical = ["payment_method", "payment_terms", "company", "industry"]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ]
    )


def build_search_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "reg",
                XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def run_training(
    csv_path: Path | None = None,
    *,
    random_state: int = 42,
    n_iter: int = 24,
    model_path: Path | None = None,
) -> tuple[RandomizedSearchCV, dict[str, float]]:
    df = load_invoices(csv_path)
    X, y_days_offset_from_due_date = build_features(df)
    preprocessor = make_preprocessor(X)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X,
        y_days_offset_from_due_date,
        random_state=random_state,
    )

    pipe = build_search_pipeline(preprocessor)
    param_distributions = {
        "reg__learning_rate": np.logspace(-2, 0, 12),
        "reg__max_depth": [3, 4, 5, 6, 8, 10],
        "reg__reg_alpha": [0.0, 0.01, 0.1, 1.0, 10.0],
        "reg__reg_lambda": [0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
    }
    n_splits = min(3, len(y_train))
    if n_splits < 2:
        raise ValueError("Not enough training rows for cross-validation.")
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    best = search.best_estimator_
    y_val_pred = best.predict(X_val)
    y_test_pred = best.predict(X_test)
    out_path = model_path or (_project_root() / "models" / "xgb_days_offset.joblib")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, out_path)

    metrics = {
        "target": "days_offset_from_due_date",
        "best_cv_neg_mae": float(search.best_score_),
        "val_mae": float(mean_absolute_error(y_val, y_val_pred)),
        "test_mae": float(mean_absolute_error(y_test, y_test_pred)),
        "val_r2": float(r2_score(y_val, y_val_pred)),
        "test_r2": float(r2_score(y_test, y_test_pred)),
        "model_path": str(out_path.resolve()),
    }
    return search, metrics


def main() -> None:
    search, metrics = run_training()
    print("Target: days_offset_from_due_date (paid_date - due_date, days)")
    print("Best params:", search.best_params_)
    print("Metrics:", metrics)
    print("Saved model:", metrics["model_path"])


if __name__ == "__main__":
    main()
