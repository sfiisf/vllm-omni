import os

from .oss_client import OssClient
from .s3_client import S3Client


class ObjectStorageClient:
    _client_types = {
        "oss": OssClient,
        "s3": S3Client,
    }

    def __init__(self):
        self.object_prefix = os.getenv("OBJECT_PREFIX", "outputs")
        self.client = self._create_client(self.object_prefix)

    @classmethod
    def _create_client(cls, object_prefix: str | None):
        obj_storage = os.getenv("OBJECT_STORAGE", "s3")
        try:
            return cls._client_types[obj_storage](object_prefix)
        except KeyError:
            raise ValueError(f"Unsupported object storage type: {obj_storage}")

    def upload_byte(self, file_name, image_byte) -> str:
        return self.client.upload_byte(file_name, image_byte)

    def upload_file(self, file_name, local_menu):
        return self.client.upload_file(file_name, local_menu)

    def object_exists(self, file_name):
        return self.client.object_exists(file_name)

    def download_file(self, file_name, local_path):
        return self.client.download_file(file_name, local_path)

    def build_url(self, target_file):
        return self.client.build_url(target_file)
