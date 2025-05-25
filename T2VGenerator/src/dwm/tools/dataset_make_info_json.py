import argparse
import dwm.tools.fs_make_info_json
import fsspec.implementations.local
import json
import os
import struct
import re


def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to make information JSON(s) for dataset to "
        "accelerate initialization.")
    parser.add_argument(
        "-dt", "--dataset-type", type=str,
        choices=["nuscenes", "waymo", "argoverse"], required=True,
        help="The dataset type.")
    parser.add_argument(
        "-s", "--split", default=None, type=str,
        help="The split, optional depending on the dataset type.")
    parser.add_argument(
        "-i", "--input-path", type=str, required=True,
        help="The path of the dataset root.")
    parser.add_argument(
        "-o", "--output-path", type=str, required=True,
        help="The path to save the information JSON file(s) on the local file "
        "system.")
    parser.add_argument(
        "-fs", "--fs-config-path", default=None, type=str,
        help="The path of file system JSON config to open the dataset.")
    return parser


if __name__ == "__main__":
    import tqdm

    parser = create_parser()
    args = parser.parse_args()

    if args.fs_config_path is None:
        fs = fsspec.implementations.local.LocalFileSystem()
    else:
        import dwm.common
        with open(args.fs_config_path, "r", encoding="utf-8") as f:
            fs = dwm.common.create_instance_from_config(json.load(f))

    if args.dataset_type == "nuscenes":
        files = [
            os.path.relpath(i, args.input_path)
            for i in fs.ls(args.input_path, detail=False)
        ]
        filtered_files = [
            i for i in files
            if (
                (args.split is None or i.startswith(args.split)) and
                i.endswith(".zip")
            )
        ]
        assert len(filtered_files) > 0, (
            "No files detected, please check the split (one of \"v1.0-mini\", "
            "\"v1.0-trainval\", \"v1.0-test\") is correct, and ensure the "
            "blob files are already converted to the ZIP format."
        )

        os.makedirs(args.output_path, exist_ok=True)
        for i in tqdm.tqdm(filtered_files):
            with fs.open("{}/{}".format(args.input_path, i)) as f:
                items = dwm.tools.fs_make_info_json.make_info_dict(
                    os.path.splitext(i)[-1], f)

            with open(
                os.path.join(
                    args.output_path, i.replace(".zip", ".info.json")),
                "w", encoding="utf-8"
            ) as f:
                json.dump(items, f)

    elif args.dataset_type == "waymo":
        import waymo_open_dataset.dataset_pb2 as waymo_pb

        files = [
            os.path.relpath(i, args.input_path)
            for i in fs.ls(args.input_path, detail=False)
            if i.endswith(".tfrecord")
        ]
        assert len(files) > 0, "No files detected."

        pattern = re.compile(
            "^segment-(?P<scene>.*)_with_camera_labels.tfrecord$")
        info_dict = {}
        for i in tqdm.tqdm(files):
            match = re.match(pattern, i)
            scene = match.group("scene")
            pt = 0
            info_list = []
            with fs.open("{}/{}".format(args.input_path, i)) as f:
                while True:
                    start = f.read(8)
                    if len(start) == 0:
                        break

                    size, = struct.unpack("<Q", start)
                    f.seek(pt + 12)
                    frame = waymo_pb.Frame()
                    frame.ParseFromString(f.read(size))
                    info_list.append([frame.timestamp_micros, size, pt + 12])

                    pt += size + 16
                    f.seek(pt)

            info_dict[scene] = info_list

        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(info_dict, f)

    elif args.dataset_type == "argoverse":
        files = [
            os.path.relpath(i, args.input_path)
            for i in fs.ls(args.input_path, detail=False)
        ]
        filtered_files = [
            i for i in files
            if (
                (args.split is None or i.startswith(args.split)) and
                i.endswith(".tar")
            )
        ]
        assert len(files) > 0, (
            "No files detected, please check the split (one of \"train\", "
            "\"val\", \"test\") is correct."
        )

        os.makedirs(args.output_path, exist_ok=True)
        for i in tqdm.tqdm(files):
            with fs.open("{}/{}".format(args.input_path, i)) as f:
                items = dwm.tools.fs_make_info_json.make_info_dict(
                    os.path.splitext(i)[-1], f, enable_tqdm=False)

            with open(
                os.path.join(
                    args.output_path, i.replace(".tar", ".info.json")),
                "w", encoding="utf-8"
            ) as f:
                json.dump(items, f)

    else:
        raise Exception("Unknown dataset type {}.".format(args.dataset_type))
