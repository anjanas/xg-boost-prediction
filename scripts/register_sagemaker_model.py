#!/usr/bin/env python3
"""
Package a local sklearn/XGBoost joblib pipeline, upload to S3, and register
a version in the Amazon SageMaker Model Registry.

Prerequisites (typical):
  - AWS credentials (env, profile, or instance role)
  - An S3 bucket in the target region
  - IAM permissions: s3:PutObject; sagemaker:CreateModelPackage, CreateModelPackageGroup,
    DescribeModelPackageGroup; optionally kms permissions on the bucket

Usage:
  export AWS_REGION=us-east-1
  export SAGEMAKER_MODEL_BUCKET=my-bucket
  python scripts/register_sagemaker_model.py \\
    --joblib-path models/xgb_days_offset.joblib \\
    --model-package-group xgb-invoice-days-offset \\
    --s3-prefix model-registry/xgb-days-offset

Optional:
  --image-uri   Override SageMaker prebuilt sklearn inference image (must include xgboost
                if your pipeline uses XGBRegressor)
  --sklearn-version  Passed to sagemaker.image_uris.retrieve (default: 1.2-1)
"""
from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError


def _default_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _build_model_tar_gz(joblib_path: Path) -> bytes:
    """
    SageMaker prebuilt sklearn inference containers load ``model.joblib`` or ``model.pkl``
    from the root of model.tar.gz.
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(joblib_path, arcname="model.joblib")
    return buf.getvalue()


def _retrieve_sklearn_inference_image(region: str, sklearn_version: str) -> str:
    try:
        from sagemaker import image_uris

        return image_uris.retrieve(
            framework="sklearn",
            region=region,
            version=sklearn_version,
            py_version="py3",
            instance_type="ml.m5.xlarge",
            image_scope="inference",
        )
    except ImportError as e:
        raise SystemExit(
            "Install the optional dependency: pip install sagemaker\n"
            "Or pass --image-uri with your inference container URI."
        ) from e


def main() -> None:
    root = _default_project_root()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--joblib-path",
        type=Path,
        default=root / "models" / "xgb_days_offset.joblib",
        help="Path to the trained pipeline joblib file",
    )
    p.add_argument(
        "--model-package-group",
        required=True,
        help="SageMaker Model Package Group name (created if missing)",
    )
    p.add_argument(
        "--s3-bucket",
        default=None,
        help="S3 bucket for model.tar.gz (default: env SAGEMAKER_MODEL_BUCKET)",
    )
    p.add_argument(
        "--s3-prefix",
        default="model-registry/xgb-days-offset",
        help="S3 key prefix (no leading slash)",
    )
    p.add_argument(
        "--region",
        default=None,
        help="AWS region (default: env AWS_REGION or AWS_DEFAULT_REGION, else session)",
    )
    p.add_argument(
        "--image-uri",
        default=None,
        help="Inference image URI (default: sklearn inference image via sagemaker SDK)",
    )
    p.add_argument(
        "--sklearn-version",
        default="1.2-1",
        help="Sklearn framework version for image_uris.retrieve",
    )
    p.add_argument(
        "--approval-status",
        default="PendingManualApproval",
        choices=["Approved", "Rejected", "PendingManualApproval"],
        help="Model approval status in the registry",
    )
    args = p.parse_args()

    bucket = args.s3_bucket or __import__("os").environ.get("SAGEMAKER_MODEL_BUCKET")
    if not bucket:
        raise SystemExit("Set --s3-bucket or environment variable SAGEMAKER_MODEL_BUCKET.")

    joblib_path = args.joblib_path.resolve()
    if not joblib_path.is_file():
        raise SystemExit(f"Joblib file not found: {joblib_path}")

    session = boto3.session.Session(region_name=args.region)
    region = session.region_name or "us-east-1"
    sm = session.client("sagemaker", region_name=region)
    s3 = session.client("s3", region_name=region)

    image_uri = args.image_uri or _retrieve_sklearn_inference_image(region, args.sklearn_version)

    artifact_key = f"{args.s3_prefix.rstrip('/')}/model.tar.gz"
    body = _build_model_tar_gz(joblib_path)
    s3.put_object(Bucket=bucket, Key=artifact_key, Body=body)
    model_data_url = f"s3://{bucket}/{artifact_key}"

    # Model Package Group (idempotent)
    try:
        sm.create_model_package_group(
            ModelPackageGroupName=args.model_package_group,
            ModelPackageGroupDescription="Invoice days-offset regressor (sklearn + XGBoost pipeline)",
        )
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code not in ("ResourceInUse", "ValidationException"):
            raise

    resp = sm.create_model_package(
        ModelPackageGroupName=args.model_package_group,
        ModelPackageDescription="Sklearn Pipeline with XGBRegressor (days offset from due date)",
        InferenceSpecification={
            "Containers": [
                {
                    "Image": image_uri,
                    "ModelDataUrl": model_data_url,
                }
            ],
            "SupportedContentTypes": ["text/csv", "application/json"],
            "SupportedResponseMIMETypes": ["text/csv", "application/json"],
        },
        ModelApprovalStatus=args.approval_status,
    )
    arn = resp["ModelPackageArn"]
    print("Uploaded:", model_data_url)
    print("ModelPackageArn:", arn)


if __name__ == "__main__":
    main()
