import boto3
from botocore.client import Config as BotoConfig
from botocore.exceptions import ClientError
from typing import IO
from src.core.config import Config
from src.core.logging import get_logger

logger = get_logger("Storage_Client")

class StorageClient:
    """
    S3 Compatible Storage Client (using boto3).
    """
    _client = None

    @staticmethod
    def get_client():
        """Initializes and returns the boto3 client singleton."""
        if StorageClient._client is None:
            try:
                StorageClient._client = boto3.client(
                    "s3",
                    endpoint_url=Config.STORAGE_ENDPOINT_URL,
                    aws_access_key_id=Config.STORAGE_ACCESS_KEY,
                    aws_secret_access_key=Config.STORAGE_SECRET_KEY,
                    config=BotoConfig(signature_version="s3v4"),
                )
                logger.info(f"Storage client initialized for endpoint: {Config.STORAGE_ENDPOINT_URL}")
            except Exception as e:
                logger.error(f"Failed to initialize storage client: {e}", exc_info=True)
                raise
        return StorageClient._client

    @staticmethod
    def upload_file_obj(file_obj: IO[bytes], object_name: str, content_type: str) -> bool:
        """
        Uploads a file-like object to the S3 bucket.
        """
        client = StorageClient.get_client()
        try:
            client.upload_fileobj(
                file_obj,
                Config.STORAGE_BUCKET_NAME,
                object_name,
                ExtraArgs={"ContentType": content_type}
            )
            logger.info(f"Successfully uploaded file to {Config.STORAGE_BUCKET_NAME}/{object_name}")
            return True
        except ClientError as e:
            logger.error(f"Failed to upload file '{object_name}': {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during upload: {e}", exc_info=True)
            return False

    @staticmethod
    def download_file_obj(object_name: str) -> bytes | None:
        """
        Downloads a file from S3 and returns its content as bytes.
        """
        client = StorageClient.get_client()
        try:
            response = client.get_object(
                Bucket=Config.STORAGE_BUCKET_NAME, 
                Key=object_name
            )
            return response["Body"].read()
        except ClientError as e:
            logger.error(f"Failed to download file '{object_name}': {e}", exc_info=True)
            return None