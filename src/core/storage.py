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
    _internal_client = boto3.client(
        "s3",
        endpoint_url=Config.STORAGE_ENDPOINT_URL,
        aws_access_key_id=Config.STORAGE_ACCESS_KEY,
        aws_secret_access_key=Config.STORAGE_SECRET_KEY,
        region_name="us-east-1", # Dummy region often required by boto3
        config=BotoConfig(signature_version="s3v4")
    )

    # Used STRICTLY to generate Presigned URLs for the browser
    _external_client = boto3.client(
        "s3",
        endpoint_url=Config.STORAGE_EXTERNAL_URL,
        aws_access_key_id=Config.STORAGE_ACCESS_KEY,
        aws_secret_access_key=Config.STORAGE_SECRET_KEY,
        region_name="us-east-1",
        config=BotoConfig(signature_version="s3v4")
    )

    @staticmethod
    def upload_file_obj(
        file_obj: IO[bytes],
        object_name: str,
        content_type: str,
        bucket_name: str = Config.STORAGE_BUCKET_AUDIO
    ) -> bool:
        """
        Uploads a file-like object to the S3 bucket.
        """
        try:
            StorageClient._internal_client.upload_fileobj(
                file_obj,
                bucket_name,
                object_name,
                ExtraArgs={"ContentType": content_type}
            )
            logger.info(
                f"Successfully uploaded file to {bucket_name}/{object_name}"
            )
            return True
        except ClientError as e:
            logger.error(f"Failed to upload file '{object_name}': {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during upload: {e}", exc_info=True
            )
            return False

    @staticmethod
    def download_file_obj(object_name: str, bucket_name: str = Config.STORAGE_BUCKET_AUDIO) -> bytes | None:
        """
        Downloads a file from S3 and returns its content as bytes.
        """
        try:
            response = StorageClient._internal_client.get_object(
                Bucket=bucket_name, 
                Key=object_name
            )
            return response["Body"].read()
        except ClientError as e:
            logger.error(f"Failed to download file '{object_name}': {e}", exc_info=True)
            return None

    @staticmethod
    def generate_presigned_url(object_name: str, bucket_name: str = Config.STORAGE_BUCKET_AUDIO, expiration=3600) -> str | None:
        """
        Generates a temporary URL for the frontend to play/download the audio directly from S3.
        This will offload bandwidth from our API server.
        """
        try:
            response = StorageClient._external_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket_name, 'Key': object_name},
                ExpiresIn=expiration
            )

            return response
        except ClientError as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            return None
        
    @staticmethod
    def delete_file(object_name: str, bucket_name: str = Config.STORAGE_BUCKET_AUDIO):
        """Deletes a file from S3."""
        try:
            StorageClient._internal_client.delete_object(Bucket=bucket_name, Key=object_name)
            return True
        except ClientError as e:
            logger.error(f"Failed to delete file S3: {e}")
            return False