import argparse
import datetime
import os
import tarfile
import zipfile


def create_parser():
    parser = argparse.ArgumentParser(
        description="The tool to convert the TAR file to the ZIP file for the "
        "capability of random access in package.")
    parser.add_argument(
        "-i", "--input-path", type=str, required=True,
        help="The input path of TAR file.")
    parser.add_argument(
        "-o", "--output-path", type=str, required=True,
        help="The output path to save the converted ZIP file.")
    parser.add_argument(
        "-e", "--extensions-to-compress", default=".bin|.json|.pcd|.txt",
        type=str, help="The extension list to compress, split by ':'.")
    return parser


def convert_content(tar_file_obj, zip_file_obj, extensions_to_compress: list):
    i = tarfile.TarInfo.fromtarfile(tar_file_obj)
    while i is not None:
        if i.isfile():
            data = tar_file_obj.extractfile(i).read()
            modified = datetime.datetime.fromtimestamp(i.mtime)
            zi = zipfile.ZipInfo(i.name, modified.timetuple()[:6])
            ext = os.path.splitext(i.name)[-1]
            compress_type = zipfile.ZIP_DEFLATED \
                if ext in extensions_to_compress else zipfile.ZIP_STORED
            zip_file_obj.writestr(zi, data, compress_type)

        i = tar_file_obj.next()


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    etc = args.extensions_to_compress.split("|")
    with tarfile.open(args.input_path) as f_in:
        with zipfile.ZipFile(args.output_path, "w") as f_out:
            convert_content(f_in, f_out, etc)
