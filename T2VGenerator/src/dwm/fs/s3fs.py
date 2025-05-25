import botocore.session
import fsspec
import io
import os
import re


class S3File(io.RawIOBase):

    @staticmethod
    def find_bucket_key(s3_path):
        """
        This is a helper function that given an s3 path such that the path is of
        the form: bucket/key
        It will return the bucket and the key represented by the s3 path
        """

        bucket_format_list = [
            re.compile(
                r"^(?P<bucket>arn:(aws).*:s3:[a-z\-0-9]*:[0-9]{12}:accesspoint[:/][^/]+)/?"
                r"(?P<key>.*)$"
            ),
            re.compile(
                r"^(?P<bucket>arn:(aws).*:s3-outposts:[a-z\-0-9]+:[0-9]{12}:outpost[/:]"
                r"[a-zA-Z0-9\-]{1,63}[/:](bucket|accesspoint)[/:][a-zA-Z0-9\-]{1,63})[/:]?(?P<key>.*)$"
            ),
            re.compile(
                r"^(?P<bucket>arn:(aws).*:s3-outposts:[a-z\-0-9]+:[0-9]{12}:outpost[/:]"
                r"[a-zA-Z0-9\-]{1,63}[/:]bucket[/:]"
                r"[a-zA-Z0-9\-]{1,63})[/:]?(?P<key>.*)$"
            ),
            re.compile(
                r"^(?P<bucket>arn:(aws).*:s3-object-lambda:[a-z\-0-9]+:[0-9]{12}:"
                r"accesspoint[/:][a-zA-Z0-9\-]{1,63})[/:]?(?P<key>.*)$"
            ),
        ]
        for bucket_format in bucket_format_list:
            match = bucket_format.match(s3_path)
            if match:
                return match.group("bucket"), match.group("key")

        s3_components = s3_path.split("/", 1)
        bucket = s3_components[0]
        s3_key = ""
        if len(s3_components) > 1:
            s3_key = s3_components[1]

        return bucket, s3_key

    def __init__(self, client, path):
        super().__init__()
        self.client = client
        self.bucket, self.key = S3File.find_bucket_key(path)
        self.p = 0
        self.head = client.head_object(Bucket=self.bucket, Key=self.key)

    def readable(self):
        return True

    def read(self, size=-1):
        read_count = min(size, self.head["ContentLength"] - self.p) \
            if size >= 0 else self.head["ContentLength"] - self.p

        if read_count == 0:
            return b""

        end = self.p + read_count - 1
        response = self.client.get_object(
            Bucket=self.bucket, Key=self.key,
            Range="bytes={}-{}".format(self.p, end))
        data = response["Body"].read()
        self.p += read_count
        return data

    def readall(self):
        return self.read(-1)

    def seek(self, offset, whence=os.SEEK_SET):
        if whence == os.SEEK_SET:
            self.p = max(0, min(offset, self.head["ContentLength"]))
        elif whence == os.SEEK_CUR:
            self.p = max(0, min(self.p + offset, self.head["ContentLength"]))
        elif whence == os.SEEK_END:
            self.p = max(
                0, min(
                    self.head["ContentLength"] + offset,
                    self.head["ContentLength"]))

        return self.p

    def seekable(self):
        return True

    def tell(self):
        return self.p

    def writable(self):
        return False


class ForkableS3FileSystem(fsspec.AbstractFileSystem):

    root_marker = ""
    protocol = "s3"
    cachable = False

    """This file system can be used to access files on the S3 service, and it
    is also compatible with the process fork under the multi-worker situation
    of the PyTorch data loader.

    Args:
        kwargs: The parameters follow the
            [Botocore confiruation](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html).
    """

    def __init__(self, **kwargs):
        # readonly, TODO: support list

        super().__init__()
        self.kwargs = kwargs
        self.client = botocore.session.get_session()\
            .create_client("s3", **kwargs)

    def reinit_if_forked(self):
        current_pid = os.getpid()
        if self._pid != current_pid:
            self.client = botocore.session.get_session()\
                .create_client("s3", **self.kwargs)
            self._pid = current_pid

    def _open(
        self, path, mode="rb", block_size=None, autocommit=True,
        cache_options=None, **kwargs,
    ):
        self.reinit_if_forked()
        return S3File(self.client, path)

    def ls(self, path, detail=True, **kwargs):
        self.reinit_if_forked()
        bucket, key = S3File.find_bucket_key(path)
        if len(key) > 0 and not key.endswith("/"):
            key = key + "/"

        # NOTE: only files are listed
        paths = []
        continuation_token = None
        while True:
            if continuation_token is None:
                response = self.client.list_objects(
                    Bucket=bucket, Delimiter="/", Prefix=key)
            else:
                response = self.client.list_objects(
                    Bucket=bucket, Delimiter="/", Prefix=key,
                    Marker=continuation_token)

            if "Contents" in response:
                for i in response["Contents"]:
                    paths.append({
                        "name": "{}/{}".format(bucket, i["Key"]),
                        "size": i["Size"],
                        "type": "file",
                        "Owner": i["Owner"],
                        "ETag": i["ETag"],
                        "LastModified": i["LastModified"]
                    })

            if response["IsTruncated"]:
                continuation_token = response["NextMarker"]
            else:
                break

        if detail:
            return sorted(paths, key=lambda i: i["name"])
        else:
            return sorted([i["name"] for i in paths])
