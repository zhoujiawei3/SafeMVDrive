import dwm.common
import fsspec.archive
import json
import os
import re
import tarfile


class CombinedTarFileSystem(fsspec.archive.AbstractArchiveFileSystem):

    root_marker = ""
    protocol = "ctar"
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
        pattern = re.compile(r"\.tar$")
        for path in paths:
            cached_info_path = re.sub(pattern, ".info.json", path)
            if enable_cached_info:
                with fs.open(cached_info_path, "r", encoding="utf-8") as f:
                    infodict = json.load(f)

                for filename, i in infodict.items():
                    _belongs_to[filename] = path

            else:
                infodict = {}
                with fs.open(path) as f:
                    with tarfile.TarFile(fileobj=f) as tf:
                        for i in tf.getmembers():
                            _belongs_to[i.name] = path
                            infodict[i.name] = [
                                i.offset_data, i.size, not i.isfile()
                            ]

            self._info[path] = dwm.common.SerializedReadonlyDict(infodict)

        self._belongs_to = dwm.common.SerializedReadonlyDict(_belongs_to)

        self.dir_cache = None
        self.fp_cache = {}

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
                blob_path = self._belongs_to[file_name]
                _, file_size, is_dir = self._info[blob_path][file_name]
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
        cache_options=None, **kwargs
    ):
        if "w" in mode:
            raise OSError("Combined TAR file system is read only.")

        if "b" not in mode:
            raise OSError(
                "Combined TAR file system only support the mode of binary.")

        if not self.exists(path):
            raise FileNotFoundError(path)

        blob_path = self._belongs_to[path]
        offset, size, is_dir = self._info[blob_path][path]
        if is_dir:
            raise FileNotFoundError(path)

        # the file points are not forkable
        current_pid = os.getpid()
        if self._pid != current_pid:
            self.fp_cache.clear()
            self._pid = current_pid

        if blob_path in self.fp_cache:
            f = self.fp_cache[blob_path]
        else:
            f = self.fp_cache[blob_path] = \
                self.fs.open(blob_path, mode, block_size, cache_options)

        return dwm.common.PartialReadableRawIO(f, offset, offset + size)
