# File System

This project uses the [fsspec](https://github.com/fsspec/filesystem_spec) to access the data from varied sources including the local file system, S3, ZIP content.


## Common usage

You can open, read, seek the file with the file system objects. Check out the [usage](https://filesystem-spec.readthedocs.io/en/latest/usage.html) document to quick start.

### Create the file system

``` Python
import fsspec.implementations.local
fs = fsspec.implementations.local.LocalFileSystem()
```

### Open and read the file

``` Python
import json
config_path = "configs/fs/local.json"
with fs.open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

from PIL import Image
image_path = "samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg"
with fs.open(image_path, "rb") as f:
    image = Image.open(f)
```


## API reference

### dwm.fs.ctar.CombinedTarFileSystem

This file system opens several TAR blobs from a given file system and provide the file access inside the TAR blobs. Please note that the required TAR file here refers only to the uncompressed format, as the TAR.GZ format does not support random access to one of the files within the blob. This file system is forkable and compatible with multi worker data loader of the PyTorch.

### dwm.fs.czip.CombinedZipFileSystem

This file system opens several ZIP blobs from a given file system and provide the file access inside the ZIP blobs. It is forkable and compatible with multi worker data loader of the PyTorch.

### dwm.fs.s3fs.ForkableS3FileSystem

This file system opens the S3 service and provide the file access on the service. It is forkable and compatible with multi worker data loader of the PyTorch.


## Configuration samples
It is easy to initialize the file system object by `dwm.common.create_instance_from_config()` with following configurations by JSON.

### Local file system

``` JSON
{
    "_class_name": "fsspec.implementations.local.LocalFileSystem"
}
```

**Relative directory on local file system**
``` JSON
{
    "_class_name": "fsspec.implementations.dirfs.DirFileSystem",
    "path": "/mnt/storage/user/wuzehuan/Downloads/data/nuscenes",
    "fs": {
        "_class_name": "fsspec.implementations.local.LocalFileSystem"
    }
}
```

### S3 file system

The parameters follow the [Botocore confiruation](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html).

``` JSON
{
    "_class_name": "dwm.fs.s3fs.ForkableS3FileSystem",
    "endpoint_url": "http://aoss-internal-v2.st-sh-01.sensecoreapi-oss.cn",
    "aws_access_key_id": "YOUR_ACCESS_KEY",
    "aws_secret_access_key": "YOUR_SECRET_KEY"
}
```

**Relative directory on S3 file system**

``` JSON
{
    "_class_name": "fsspec.implementations.dirfs.DirFileSystem",
    "path": "users/wuzehuan/data/nuscenes",
    "fs": {
        "_class_name": "dwm.fs.s3fs.ForkableS3FileSystem",
        "endpoint_url": "http://aoss-internal-v2.st-sh-01.sensecoreapi-oss.cn",
        "aws_access_key_id": "YOUR_ACCESS_KEY",
        "aws_secret_access_key": "YOUR_SECRET_KEY"
    }
}
```

**Retry options on S3 file system**

``` JSON
{
    "_class_name": "dwm.fs.s3fs.ForkableS3FileSystem",
    "endpoint_url": "http://aoss-internal-v2.st-sh-01.sensecoreapi-oss.cn",
    "aws_access_key_id": "YOUR_ACCESS_KEY",
    "aws_secret_access_key": "YOUR_SECRET_KEY",
    "config": {
        "_class_name": "botocore.config.Config",
        "retries": {
            "max_attempts": 8
        }
    }
}
```
