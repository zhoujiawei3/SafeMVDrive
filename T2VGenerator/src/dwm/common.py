import bisect
import importlib
import io
import numpy as np
import os
import pickle


class PartialReadableRawIO(io.RawIOBase):
    def __init__(
        self, base_io_object: io.RawIOBase, start: int, end: int,
        close_with_this_object: bool = False
    ):
        super().__init__()
        self.base_io_object = base_io_object
        self.p = self.start = start
        self.end = end
        self.close_with_this_object = close_with_this_object
        self.base_io_object.seek(start)

    def close(self):
        if self.close_with_this_object:
            self.base_io_object.close()

    @property
    def closed(self):
        return self.base_io_object.closed if self.close_with_this_object \
            else False

    def readable(self):
        return self.base_io_object.readable()

    def read(self, size=-1):
        read_count = min(size, self.end - self.p) \
            if size >= 0 else self.end - self.p
        data = self.base_io_object.read(read_count)
        self.p += read_count
        return data

    def readall(self):
        return self.read(-1)

    def seek(self, offset, whence=os.SEEK_SET):
        if whence == os.SEEK_SET:
            p = max(0, min(self.end - self.start, offset))
        elif whence == os.SEEK_CUR:
            p = max(
                0, min(self.end - self.start, self.p - self.start + offset))
        elif whence == os.SEEK_END:
            p = max(
                0, min(self.end - self.start, self.end - self.start + offset))

        self.p = self.base_io_object.seek(self.start + p, os.SEEK_SET)
        return self.p

    def seekable(self):
        return self.base_io_object.seekable()

    def tell(self):
        return self.p - self.start

    def writable(self):
        return False


class ReadonlyDictIndices:
    def __init__(self, base_dict_keys):
        sorted_table = sorted(
            enumerate(base_dict_keys), key=lambda i: i[1])
        self.sorted_keys = [i[1] for i in sorted_table]
        self.key_indices = np.array([i[0] for i in sorted_table], np.int64)

    def __len__(self):
        return len(self.sorted_keys)

    def __contains__(self, key):
        i = bisect.bisect_left(self.sorted_keys, key)
        in_range = i >= 0 and i < len(self.sorted_keys)
        return in_range and self.sorted_keys[i] == key

    def __getitem__(self, key):
        i = bisect.bisect_left(self.sorted_keys, key)
        if i < 0 or i >= len(self.sorted_keys) or self.sorted_keys[i] != key:
            raise KeyError("{} not found".format(key))

        return self.key_indices[i]

    def get_all_indices(self, key):
        i0 = bisect.bisect_left(self.sorted_keys, key)
        i1 = bisect.bisect_right(self.sorted_keys, key)
        return [self.key_indices[i] for i in range(i0, i1)] if i1 > i0 else []


class SerializedReadonlyList:
    """A list to prevent memory divergence accessed by forked process.
    """

    def __init__(self, items: list):
        serialized_items = [pickle.dumps(i) for i in items]
        self.offsets = np.cumsum([len(i) for i in serialized_items])
        self.blob = b''.join(serialized_items)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index: int):
        start = 0 if index == 0 else self.offsets[index - 1]
        end = self.offsets[index]
        return pickle.loads(self.blob[start:end])


class SerializedReadonlyDict:
    """A dict to prevent memory divergence accessed by forked process.
    """

    def __init__(self, base_dict: dict):
        self.indices = ReadonlyDictIndices(base_dict.keys())
        self.values = SerializedReadonlyList(base_dict.values())

    def __len__(self):
        return len(self.indices)

    def __contains__(self, key):
        return key in self.indices

    def __getitem__(self, key):
        return self.values[self.indices[key]]

    def keys(self):
        return self.indices.sorted_keys


def get_class(class_name: str):
    if "." in class_name:
        i = class_name.rfind(".")
        module_name = class_name[:i]
        class_name = class_name[i+1:]

        module_type = importlib.import_module(module_name, package=None)
        class_type = getattr(module_type, class_name)
    elif class_name in globals():
        class_type = globals()[class_name]
    else:
        raise RuntimeError("Failed to find the class {}.".format(class_name))

    return class_type


def create_instance(class_name: str, **kwargs):
    class_type = get_class(class_name)
    return class_type(**kwargs)


def create_instance_from_config(_config: dict, level: int = 0, **kwargs):
    if isinstance(_config, dict):
        if "_class_name" in _config:
            args = instantiate_config(_config, level)
            if level == 0:
                args.update(kwargs)

            if _config["_class_name"] == "get_class":
                return get_class(**args)
            else:
                return create_instance(_config["_class_name"], **args)

        else:
            return instantiate_config(_config, level)

    elif isinstance(_config, list):
        return [create_instance_from_config(i, level + 1) for i in _config]
    else:
        return _config


def instantiate_config(_config: dict, level: int = 0):
    return {
        k: create_instance_from_config(v, level + 1)
        for k, v in _config.items() if k != "_class_name"
    }


def get_state(key: str):
    return global_state[key]


global_state = {}
