"""
Package the model code into model.tar.gz and upload to S3 for SageMaker deployment.

Run from the project root:
    python endpoint/package_model.py
"""
import logging
import os
import tarfile

import boto3

from config import S3_BUCKET, S3_EMBEDDINGS_PREFIX

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# Paths (relative to project root) to bundle into model.tar.gz
INCLUDE_PATHS = [
    "config.py",
    "requirements.txt",
    "indexing/",
    "matching/",
    "endpoint/inference.py",
]

OUTPUT_TAR = "/tmp/model.tar.gz"
S3_MODEL_KEY = f"{S3_EMBEDDINGS_PREFIX}model.tar.gz"


def package_model() -> None:
    """
    Create model.tar.gz from project source files and upload to S3.

    SageMaker will unpack this archive into model_dir when the endpoint starts.
    """
    logger.info("Creating %s", OUTPUT_TAR)

    def _exclude(info):
        if "__pycache__" in info.name or info.name.endswith(".pyc"):
            return None
        return info

    with tarfile.open(OUTPUT_TAR, "w:gz") as tar:
        for path in INCLUDE_PATHS:
            if os.path.exists(path):
                tar.add(path, filter=_exclude)
                logger.info("  added: %s", path)
            else:
                logger.warning("  skipped (not found): %s", path)

    logger.info("Uploading to s3://%s/%s", S3_BUCKET, S3_MODEL_KEY)
    s3 = boto3.client("s3")
    s3.upload_file(OUTPUT_TAR, S3_BUCKET, S3_MODEL_KEY)
    logger.info("Upload complete.")


if __name__ == "__main__":
    package_model()
