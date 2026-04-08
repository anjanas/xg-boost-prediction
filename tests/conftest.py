from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def sample_invoices() -> pd.DataFrame:
    """Minimal invoice rows matching columns expected by build_features."""
    return pd.DataFrame(
        {
            "invoice_id": ["INV-1001", "INV-1002"],
            "creation_date": ["2026-01-10", "2026-02-01"],
            "customer": ["James Patel", "Unknown Person"],
            "company": ["Riverbend Wholesale Supply Co.", "MetroMart National Retail Group"],
            "item_description": ["Bolts 10pk", "Hose 50ft"],
            "unit_price": [12.5, 8.0],
            "quantity": [100, 50],
            "invoice_amount": [1250.0, 400.0],
            "due_date": ["2026-01-31", "2026-03-01"],
            "paid_date": ["2026-02-02", "2026-03-05"],
            "payment_method": ["credit card", "cheque"],
            "payment_terms": ["net 30", "net 90"],
        }
    )
