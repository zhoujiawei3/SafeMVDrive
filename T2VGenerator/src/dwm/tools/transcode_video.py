import argparse
import json
import os
import subprocess


def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to convert videos for better performance.")
    parser.add_argument(
        "-c", "--config-path", type=str, required=True,
        help="The config path.")
    parser.add_argument(
        "-i", "--input-path", type=str, required=True,
        help="The input path of videos.")
    parser.add_argument(
        "-o", "--output-path", type=str, required=True,
        help="The output path to save the merged dict file.")
    parser.add_argument("-f", "--range-from", type=int, default=0)
    parser.add_argument("-t", "--range-to", type=int, default=-1)
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    files = os.listdir(args.input_path)
    range_to = range_to = len(files) if args.range_to == -1 else args.range_to
    print(
        "Dataset count: {}, processing range {} - {}".format(
            len(files), args.range_from, range_to))
    actual_files = files[args.range_from:range_to]
    print(actual_files)
    os.makedirs(args.output_path, exist_ok=True)
    for i in actual_files:
        if not os.path.isfile(os.path.join(args.input_path, i)):
            continue

        if "reformat" in config:
            output_file = os.path.splitext(i)[0] + config["reformat"]
        else:
            output_file = i

        subprocess.run([
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-i",
            os.path.join(args.input_path, i),
            *config["ffmpeg_args"],
            os.path.join(args.output_path, output_file)
        ], stdout=subprocess.PIPE, check=True)
