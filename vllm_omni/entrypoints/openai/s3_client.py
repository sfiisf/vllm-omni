import configparser
import io
import os
import time
from enum import Enum
from functools import cached_property, wraps
from pathlib import Path
from threading import RLock
from typing import cast, NoReturn
from urllib.parse import urljoin, urlparse, urlunparse

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from furl import furl
from vllm.logger import init_logger
from types_boto3_s3.client import S3Client as Boto3S3Client

logger = init_logger(__name__)

class CredSource(Enum):
    ENV = "env"
    FILE = "file"


FILES_ENDPOINT = {
    "files.cn-shanghai-1.siliconflow.cn",
    "s3.siliconflow.cn",
    "s3.6scloud.com",
    "uat-files.siliconflow.cn",
}


class S3Client:
    def __init__(
        self,
        menu: str = os.getenv("OBJECT_PREFIX", ""),
        *,
        bucket_name: str = os.getenv("S3_BUCKET_NAME", ""),
        s3_endpoint_url: str = os.getenv("S3_ENDPOINT", ""),
        s3_region_name: str = os.getenv("S3_REGION_NAME", ""),
        s3_access_key: str = os.getenv("S3_ACCESS_KEY", ""),
        s3_secret_key: str = os.getenv("S3_SECRET_KEY", ""),
        file_expire_time: int = int(os.getenv("S3_EXPIRE_SECONDS", "3600")),
        enable_signed_url: bool = bool(os.getenv("S3_ENABLE_PRESIGNED_URL", "True").lower() == 'true'),
        credentials_source: str | CredSource = CredSource(os.getenv("S3_CREDENTIALS_SOURCE", "file")),
        cred_reload_interval: float = float(os.getenv("S3_CREDENTIALS_RELOAD_SECONDS", "7200")),
    ):
        """Initialize S3Client with configuration from environment variables."""

        # s3 config
        self.s3_endpoint_url = s3_endpoint_url
        self.s3_region_name = s3_region_name
        self.bucket_name = bucket_name
        self.object_prefix = menu
        self.file_expire_time = file_expire_time
        self.s3_access_key = s3_access_key
        self.s3_secret_key = s3_secret_key
        self.enable_signed_url = enable_signed_url

        self.s3_config = {}
        self.s3_client: Boto3S3Client
        if isinstance(credentials_source, str):
            credentials_source = CredSource(credentials_source)
        self.credentials_source = credentials_source
        # The credential file on FaaS is updated every **three** hours,
        # and the token provided in each updated credential file expires after 48 hours.
        # Therefore, the S3 client needs to refresh periodically
        # to obtain a valid token from the latest credential file.
        self.cred_reload_interval: float = cred_reload_interval

        self._lock = RLock()

        self._init_config()
        self.validate_bucket()

    # https://stackoverflow.com/questions/63724485/how-to-refresh-the-boto3-credentials-when-python-script-is-running-indefinitely
    def _read_aws_credentials(self) -> configparser.ConfigParser | None:
        config: configparser.ConfigParser = configparser.ConfigParser()
        env_cred_file = os.environ.get("AWS_SHARED_CREDENTIALS_FILE")
        if env_cred_file is None:
            raise FileNotFoundError(
                "Environment variable AWS_SHARED_CREDENTIALS_FILE not set when load s3 credential from file"
            )
        try:
            config.read(env_cred_file)
        except Exception as e:
            logger.warning(f"Parse config from AWS_SHARED_CREDENTIALS_FILE error, reason: {e}")
        return config

    def _generate_client(self):
        self.s3_client = cast(
            Boto3S3Client,
            boto3.client(
                "s3",
                **self.s3_config,
                config=Config(
                    signature_version="s3v4",
                    # TODO: use virtual to replace path
                    s3={"addressing_style": "path"},
                ),
            ),
        )

    def _init_config(self):
        if self.s3_endpoint_url:
            self.s3_config["endpoint_url"] = self.s3_endpoint_url
        if self.s3_region_name:
            self.s3_config["region_name"] = self.s3_region_name

        if self.credentials_source == CredSource.ENV:
            if self.s3_access_key is None or self.s3_secret_key is None:
                raise ValueError(
                    "S3 access key and secret key must be provided when using environment variable credentials"
                )
            logger.debug("Created S3 client from env credential")
            self.s3_config["aws_access_key_id"] = self.s3_access_key
            self.s3_config["aws_secret_access_key"] = self.s3_secret_key
            self._generate_client()
        elif self.credentials_source == CredSource.FILE:
            logger.debug("Created S3 client from file credential")
            self.cred_expire_time: float = -1
            self._update_client()
        else:
            raise ValueError(f"Unsupported credentials source: {self.credentials_source}")

    def _update_config(self):
        credential_config = self._read_aws_credentials()
        if credential_config is None:
            return
        logger.debug("Loaded credential file expired, reload credential file")
        self.s3_config["aws_access_key_id"] = credential_config.get("default", "aws_access_key_id")
        self.s3_config["aws_secret_access_key"] = credential_config.get("default", "aws_secret_access_key")
        aws_session_token = credential_config.get("default", "aws_session_token", fallback=None)
        if aws_session_token:
            self.s3_config["aws_session_token"] = aws_session_token
        else:
            self.s3_config.pop("aws_session_token", None)

    def _update_client(self):
        if self.credentials_source == CredSource.ENV:
            return
        with self._lock:
            if time.time() > self.cred_expire_time:
                self._update_config()
                self._generate_client()
                self.cred_expire_time = time.time() + self.cred_reload_interval

    @cached_property
    def _is_using_files(self) -> bool:
        return furl(self.s3_endpoint_url).host in FILES_ENDPOINT

    @staticmethod
    def refresh_credentials(fn):
        @wraps(fn)
        def wrapper(self: "S3Client", *args, **kwargs):
            self._update_client()
            result = fn(self, *args, **kwargs)
            return result

        return wrapper

    def build_object_key(self, file_name: str) -> str:
        if self.object_prefix is None:
            return file_name
        return f"{self.object_prefix.rstrip('/')}/{file_name}"

    def validate_bucket(self) -> None | NoReturn:
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            return
        # files not supports head bucket yet
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            match error_code, self._is_using_files:
                case "404", True:
                    logger.info("Using Siliconflow Files, head_bucket is not supported yet. " "Fallback to list_bucket")
                case "404", False:
                    logger.error(f"Bucket '{self.bucket_name}' doesn't exist")
                    raise ValueError(f"Bucket '{self.bucket_name}' doesn't exist")
                case "403", False:
                    logger.error(f"No permission to access bucket '{self.bucket_name}'")
                    raise ValueError(f"No permission to access bucket '{self.bucket_name}'")
                case "500", True:
                    logger.warning(f"Head bucket '{self.bucket_name}' failed, reason: {e}")
                case _:
                    logger.error(f"Can't access bucket '{self.bucket_name}', reason: {e}")
                    raise RuntimeError(f"Can't access bucket '{self.bucket_name}', reason: {e}")
        try:
            list_buckets_output = self.s3_client.list_buckets()
            buckets = {x["Name"] for x in list_buckets_output["Buckets"]}
        except ClientError as e:
            logger.error(f"list buckets failed, reason: {str(e)}")
            raise ValueError(f"list buckets failed, reason: {str(e)}")
        except Exception as e:
            logger.error(f"list buckets failed, reason: {str(e)}")
            raise
        if self.bucket_name not in buckets:
            raise ValueError(f"Bucket '{self.bucket_name}' doesn't exist")

    @refresh_credentials
    def upload_byte(self, file_name: str, bytes: bytes) -> str:
        if self.s3_client is None:
            raise ValueError("s3 client not ready")
        object_name = self.build_object_key(file_name)
        max_retries = 3
        error = None

        for retry in range(max_retries):
            try:
                with io.BytesIO(bytes) as fileobj:
                    fileobj.seek(0)
                    self.s3_client.upload_fileobj(fileobj, self.bucket_name, object_name)
                logger.debug(f"File {file_name} uploaded to {self.bucket_name}/{object_name}")
                return self.generate_presigned_url(object_name, self.file_expire_time)
            except Exception as e:
                logger.warning(f"s3 client upload byte failed, error: {str(e)} after {retry} times attempt")
                error = e
        logger.error(f"s3 client upload byte failed, error: {str(error)} after {max_retries} times attempt")
        raise ValueError(f"s3 client upload byte failed, error: {str(error)}")

    @refresh_credentials
    def upload_file(self, file_name: str, local_menu: str) -> str:
        if self.s3_client is None:
            raise ValueError("s3 client not ready")
        # TODO: upload_file is not called
        file_path = f"{local_menu}/{file_name}"
        object_name = self.build_object_key(file_name)
        max_retries = 3

        error = None
        for retry in range(max_retries):
            try:
                self.s3_client.upload_file(file_path, self.bucket_name, object_name)
                logger.debug(f"File {file_path} uploaded to {self.bucket_name}/{object_name}")
                return self.generate_presigned_url(object_name, self.file_expire_time)
            except Exception as e:
                logger.warning(f"s3 client upload file failed, error: {str(e)} after {retry} times attempt")
                error = e
        logger.error(f"s3 client upload file failed, error: {str(error)} after {max_retries} times attempt")
        raise ValueError(f"s3 client upload file failed, error: {str(error)}")

    def generate_presigned_url(self, object_file: str, expired_in_sec: int) -> str:
        if self.s3_client is None:
            raise ValueError("s3 client not ready")

        effective_signed_url = self.enable_signed_url

        try:
            endpoint = (self.s3_endpoint_url or "").strip()
            if not endpoint:
                final_url = urljoin(f"{self.bucket_name}/", object_file.lstrip("/"))
            else:
                base = urljoin(f"{endpoint}/", f"{self.bucket_name}/")
                final_url = urljoin(base, object_file.lstrip("/"))

            if effective_signed_url:
                raw_signed_url = self.s3_client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": self.bucket_name, "Key": object_file},
                    ExpiresIn=expired_in_sec,
                )
                parsed_signed_url = urlparse(raw_signed_url)
                if not urlparse(final_url).netloc:
                    final_url = urlunparse(parsed_signed_url._replace(query=""))
                final_url = urlunparse(urlparse(final_url)._replace(query=parsed_signed_url.query))

            return self.convert_http_to_https(final_url)

        except Exception as e:
            logger.error(f"Generate URL failed, error: {str(e)}")
            raise ValueError(f"Generate URL failed: {str(e)}")

    def convert_http_to_https(self, url):
        if url.startswith("http://"):
            return url.replace("http://", "https://", 1)
        return url

    def object_exists(self, object_name: str) -> bool:
        object_name = self.build_object_key(object_name)
        if self.s3_client is None:
            raise ValueError("s3 client not ready")
        try:
            logger.debug(f"get object meta for object file: {object_name}")
            self.s3_client.head_object(Bucket=self.bucket_name, Key=object_name)
            return True
        except ClientError as e:
            response = e.response
            if response.get("Error", {}).get("Code") == "404":
                logger.debug(f"get object meta failed for object file: {object_name}, 404 not found")
                return False
            else:
                logger.error(f"get object meta failed for object file: {object_name}, error: {str(e)}")
                raise ValueError(f"get object meta failed for object file: {object_name}, error: {str(e)}",
                )
        except Exception as e:
            raise ValueError(f"get object meta failed for object file: {object_name}, error: {str(e)}",
            )

    def download_file(self, file_name: str, local_path: str) -> None:
        object_name = self.build_object_key(file_name)
        if self.s3_client is None:
            raise ValueError("s3 client not ready")
        try:
            logger.debug(f"download object file {file_name} to path {local_path}")
            self.s3_client.download_file(self.bucket_name, object_name, local_path)
        except Exception as e:
            logger.error(f"download object file {file_name} to path {local_path} failed, error: {str(e)}")
            raise ValueError(f"download object file {file_name} to path {local_path} failed, error: {str(e)}",
            )
