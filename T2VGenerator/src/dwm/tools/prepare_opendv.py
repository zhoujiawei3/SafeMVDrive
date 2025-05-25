import json
from tqdm import tqdm
from datasets import load_dataset
import argparse

def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to make information JSON(s) for \
        official opendv annotations")
    parser.add_argument(
        "--meta-path", type=str, required=True,
        help="The path to official video metas of OpenDV-2K.")
    parser.add_argument(
        "-o", "--output-path", type=str, required=True,
        help="The path to save the information JSON file(s) on the local file "
        "system.")
    return parser

if __name__ == "__main__":    
    parser = create_parser()
    args = parser.parse_args()

    ds = load_dataset("OpenDriveLab/OpenDV-YouTube-Language")
    split = None

    with open(args.meta_path, "r", encoding="utf-8") as f:
        meta_dict = {
            i["videoid"]: i
            for i in json.load(f)
            if split is None or i["split"] == split and
            i["videoid"] not in ignore_list
        }

    new_image_descriptions = dict()
    for sp in ["train", "validation"]:
        with tqdm(
            ds[sp],
            desc=f"Prepare {sp}",
        ) as pbar:
            for frame_annos in pbar:
                k1 = frame_annos['folder'].split('/')[-1]
                k2 = int(frame_annos['first_frame'].split('.')[0])
                v = (frame_annos['blip'], frame_annos['cmd'])
                if k1 not in meta_dict:
                    continue
                start_discard = meta_dict[k1]["start_discard"]
                default_fps, time_base = 10, 0.001
                t = int((k2/default_fps+start_discard)/time_base)
                new_image_descriptions[f"{k1}.{t}"] = {
                    "image_description": v[0],
                    "action": v[1]
                }
    with open(
        args.output_path, "w", encoding="utf-8"
    ) as f:
        json.dump(new_image_descriptions, f)