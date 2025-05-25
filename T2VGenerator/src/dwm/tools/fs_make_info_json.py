import argparse
import fsspec.implementations.local
import json
import os
import tarfile
import zipfile


def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to make information JSON for TAR or ZIP files "
        "to accelerate file system initialization.")
    parser.add_argument(
        "-i", "--input-path", type=str, required=True,
        help="The path of the TAR or ZIP file.")
    parser.add_argument(
        "-o", "--output-path", type=str, required=True,
        help="The path to save the information JSON file on the local file "
        "system.")
    parser.add_argument(
        "-fs", "--fs-config-path", default=None, type=str,
        help="The path of file system JSON config to open the input file.")
    return parser


def enum_tar_members(tarfile):
    tarfile._check()
    while True:
        tarinfo = tarfile.next()
        if tarinfo is None:
            break

        yield tarinfo


def make_info_dict(ext, fobj, enable_tqdm: bool = True):
    if ext == ".zip":
        with zipfile.ZipFile(fobj) as zf:
            return {
                i.filename: [i.header_offset, i.file_size, i.is_dir()]
                for i in zf.infolist()
            }
    elif ext == ".tar":
        with tarfile.TarFile(fileobj=fobj) as tf:
            tar_member_generator = enum_tar_members(tf)
            if enable_tqdm:
                # Since TAR files do not have centralized index information,
                # the scanning process will take a longer time.
                import tqdm
                tar_member_generator = tqdm.tqdm(tar_member_generator)

            return {
                i.name: [i.offset_data, i.size, not i.isfile()]
                for i in tar_member_generator
            }
    else:
        raise Exception("Unknown format {}.".format(ext))


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    if args.fs_config_path is None:
        fs = fsspec.implementations.local.LocalFileSystem()
    else:
        import dwm.common
        with open(args.fs_config_path, "r", encoding="utf-8") as f:
            fs = dwm.common.create_instance_from_config(json.load(f))

    with fs.open(args.input_path, "rb") as f:
        items = make_info_dict(os.path.splitext(args.input_path)[-1], f)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(items, f)
