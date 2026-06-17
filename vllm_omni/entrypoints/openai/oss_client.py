import os
import oss2
from vllm.logger import init_logger

logger = init_logger(__name__)

_OSS_ENDPOINT = os.getenv("OSS_ENDPOINT", None)
_OSS_BUCKET_NAME = os.getenv("OSS_BUCKET_NAME", None)
_OSS_ACCESS_KEY_ID = os.getenv("OSS_ACCESS_KEY_ID", None)
_OSS_ACCESS_KEY_SECRET = os.getenv("OSS_ACCESS_KEY_SECRET", None)
_OSS_ExpiresIn_TIME = int(os.getenv("OSS_ExpiresIn_TIME", 3600))
_OSS_ENABLE_SIGNED_URL = bool(os.getenv("OSS_ENABLE_SIGNED_URL", "True").lower() == "true")

class OssClient:
    def __init__(
        self,
        object_prefix: str | None = None,
        *,
        bucket_name: str = _OSS_BUCKET_NAME,
        endpoint: str = _OSS_ENDPOINT,
        access_key_id: str = _OSS_ACCESS_KEY_ID,
        access_key_secret: str = _OSS_ACCESS_KEY_SECRET,
        expired_in_sec: int = _OSS_ExpiresIn_TIME,
    ):
        # TODO: 错误处理
        try:
            self.endpoint = endpoint
            self.bucket_name = bucket_name
            self.expired_in_sec = expired_in_sec
            self.object_prefix = object_prefix

            auth = oss2.Auth(access_key_id, access_key_secret)
            bucket = oss2.Bucket(auth, self.endpoint, self.bucket_name, connect_timeout=5)
            self.bucket = bucket
        except Exception as e:
            logger.error(f"OssClient init got exception: {str(e)}")
            raise

    def build_object_key(self, file_name: str) -> str:
        if self.object_prefix is None:
            return file_name
        return f"{self.object_prefix}/{file_name}"

    def upload_file(self, file_name, local_menu) -> str:
        local_file = "{}/{}".format(local_menu, file_name)
        target_file = self.build_object_key(file_name)
        retries = 3  # 设置最大尝试次数

        for attempt in range(retries):
            try:
                result = self.bucket.put_object_from_file(target_file, local_file)
                if result.status == 200:
                    return self.build_url(target_file)
                else:
                    logger.warning("oss upload failed with status: {}, cnt: {}".format(result.status, attempt))
            except Exception as e:
                logger.warning("oss upload exception: {}, cnt: {}".format(e, attempt))
            # 如果重试次数用完，返回 None
            if attempt == retries - 1:
                raise ValueError(f"oss uploading {file_name} failed")

        # will never execute
        raise ValueError("oss upload failed")

    # TODO: add timeout
    def upload_byte(self, file_name, image_stream) -> str:
        target_file = self.build_object_key(file_name)
        retries = 3  # 设置最大尝试次数

        for attempt in range(retries):
            try:
                result = self.bucket.put_object(target_file, image_stream)
                if result.status == 200:
                    return self.build_url(target_file)
                else:
                    logger.warning("oss upload failed with status: {}, cnt: {}".format(result.status, attempt))
            except Exception as e:
                logger.warning("oss upload exception: {}, cnt: {}".format(e, attempt))
            # 如果重试次数用完，返回 None
            if attempt == retries - 1:
                raise ValueError(f"oss uploading file {file_name} failed")
        raise ValueError("oss upload failed")

    def build_url(self, target_file, expired_in_sec=None):
        if _OSS_ENABLE_SIGNED_URL:
            return self.sign_url(target_file, expired_in_sec or self.expired_in_sec)
        else:
            return f"https://{self.bucket_name}.{self.endpoint}/{target_file}"

    def object_exists(self, file_name):
        target_file = self.build_object_key(file_name)
        try:
            self.bucket.get_object_meta(target_file)
            return True
        except oss2.exceptions.NoSuchKey:
            return False
        except Exception as e:
            logger.warning(f"Object exists check error: {e}")
            return False

    def download_file(self, file_name, local_path):
        target_file = self.build_object_key(file_name)
        try:
            result = self.bucket.get_object_to_file(target_file, local_path)
            if result.status == 200:
                logger.debug(f"OSS downloaded {target_file} to {local_path}")
            else:
                logger.warning(f"OSS download failed: status={result.status}")
        except oss2.exceptions.NoSuchKey:
            logger.warning(f"OSS object not found for download: {target_file}")
        except Exception as e:
            raise ValueError(f"OSS download failed: {e}")

    def sign_url(self, target_file, expired_in_sec) -> str:
        try:
            signed_url = self.bucket.sign_url("GET", target_file, expired_in_sec)
            logger.debug("signed_url: {}".format(signed_url))
            return convert_http_to_https(signed_url)
        except Exception as e:
            logger.error("sign oss url fail. error: {}".format(e))
            raise ValueError(f"oss sign url failed. error: {str(e)}")


def convert_http_to_https(url):
    if url.startswith("http://"):
        return url.replace("http://", "https://", 1)
    return url
