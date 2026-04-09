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

## AWS (CLI + credentials)

Install the CLI (macOS):

```bash
brew install awscli
```

Put credentials in `~/.aws/credentials` (`[default]`, `aws_access_key_id`, `aws_secret_access_key`) and set default region in `~/.aws/config`, or run `aws configure`. Check:

```bash
aws sts get-caller-identity
```

For packaging the joblib artifact to S3 and registering in SageMaker Model Registry, see `scripts/register_sagemaker_model.py` (needs `boto3`, optional `sagemaker` for the default inference image URI, plus `SAGEMAKER_MODEL_BUCKET` and IAM permissions for S3 + model registry).
