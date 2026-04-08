# xg-boost-prediction

## Training

From the project root, with your Python environment having the dependencies used by `src/xgboost_training.py` (e.g. `pandas`, `scikit-learn`, `xgboost`, `joblib`):

```bash
python src/xgboost_training.py
```

## Saved model

After training, the fitted pipeline is written to:

`models/xgb_days_offset.joblib`

The training script also prints this path in the metrics output as `model_path`.
