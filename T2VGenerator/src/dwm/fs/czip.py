import dwm.common
import fsspec
import fsspec.archive
import io
import json
import os
import re
import struct
import zipfile
import zlib


class CombinedZipFileSystem(fsspec.archive.AbstractArchiveFileSystem):

    """This file system can be used to access files inside ZIP files, and it is
    also compatible with the process fork under the multi-worker situation of
    the PyTorch data loader.

    Args:
        paths (list): The path list of ZIP files to open.
        fs (fsspec.AbstractFileSystem or None): The file system to open the ZIP
            blobs, or local file system as default.
        enable_cached_info (bool): Load the byte offset of ZIP entries from
            cached info file with the extension `.info.json` to accelerate
            opening large ZIP files.
    """

    root_marker = ""
    protocol = "czip"
    cachable = False

    def __init__(
        self, paths: list, fs=None,
        enable_cached_info: bool = False, **kwargs
    ):
        super().__init__(**kwargs)

        self.fs = fsspec.implementations.local.LocalFileSystem() \
            if fs is None else fs
        self.paths = paths

        _belongs_to = {}
        self._info = {}
        pattern = re.compile(r"\.zip$")
        for path in paths:
            cached_info_path = re.sub(pattern, ".info.json", path)
            if enable_cached_info:
                with fs.open(cached_info_path, "r", encoding="utf-8") as f:
                    infodict = json.load(f)

                for filename, i in infodict.items():
                    _belongs_to[filename] = path

            else:
                with fs.open(path) as f:
                    with zipfile.ZipFile(f) as zf:
                        infodict = {}
                        for i in zf.infolist():
                            _belongs_to[i.filename] = path
                            infodict[i.filename] = [
                                i.header_offset, i.file_size, i.is_dir()
                            ]

            self._info[path] = dwm.common.SerializedReadonlyDict(infodict)

        self._belongs_to = dwm.common.SerializedReadonlyDict(_belongs_to)

        self.dir_cache = None
        self.fp_cache = {}

    @classmethod
    def _strip_protocol(cls, path):
        # zip file paths are always relative to the archive root
        return super()._strip_protocol(path).lstrip("/")

    def close(self):
        # the file points are not forkable
        current_pid = os.getpid()
        if self._pid != current_pid:
            self.fp_cache.clear()
            self._pid = current_pid

        for fp in self.fp_cache.values():
            fp.close()

        self.fp_cache.clear()

    def _get_dirs(self):
        if self.dir_cache is None:
            self.dir_cache = {
                dirname.rstrip("/"): {
                    "name": dirname.rstrip("/"),
                    "size": 0,
                    "type": "directory",
                }
                for dirname in self._all_dirnames(self._belongs_to.keys())
            }

            for file_name in self._belongs_to.keys():
                zip_path = self._belongs_to[file_name]
                _, file_size, is_dir = self._info[zip_path][file_name]
                f = {
                    "name": file_name.rstrip("/"),
                    "size": file_size,
                    "type": "directory" if is_dir else "file",
                }
                self.dir_cache[f["name"]] = f

    def exists(self, path, **kwargs):
        return path in self._belongs_to

    def _open(
        self, path, mode="rb", block_size=None, autocommit=True,
        cache_options=None, **kwargs,
    ):
        if "w" in mode:
            raise OSError("Combined stateless ZIP file system is read only.")

        if "b" not in mode:
            raise OSError(
                "Combined stateless ZIP file system only support the mode of "
                "binary.")

        if not self.exists(path):
            raise FileNotFoundError(path)

        zip_path = self._belongs_to[path]
        header_offset, size, is_dir = self._info[zip_path][path]
        if is_dir:
            raise FileNotFoundError(path)

        # the file points are not forkable
        current_pid = os.getpid()
        if self._pid != current_pid:
            self.fp_cache.clear()
            self._pid = current_pid

        if zip_path in self.fp_cache:
            f = self.fp_cache[zip_path]
        else:
            f = self.fp_cache[zip_path] = \
                self.fs.open(zip_path, mode, block_size, cache_options)

        f.seek(header_offset)
        fh = struct.unpack(zipfile.structFileHeader, f.read(30))
        method = fh[zipfile._FH_COMPRESSION_METHOD]
        offset = header_offset + 30 + fh[zipfile._FH_FILENAME_LENGTH] + \
            fh[zipfile._FH_EXTRA_FIELD_LENGTH]

        if method == zipfile.ZIP_STORED:
            return dwm.common.PartialReadableRawIO(f, offset, offset + size)

        elif method == zipfile.ZIP_DEFLATED:
            f.seek(offset)
            data = f.read(size)
            result = io.BytesIO(zlib.decompress(data, -15))
            return result

        else:
            raise NotImplementedError("The compression method is unsupported")
