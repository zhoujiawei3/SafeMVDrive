import argparse
import torch.distributed
import dwm.common
import dwm.utils.preview
import json
import os
import torch
import torchvision


def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to finetune a stable diffusion model to the "
        "driving dataset.")
    parser.add_argument(
        "-c", "--config-path", type=str, required=True,
        help="The config to load the train model and dataset.")
    parser.add_argument(
        "-o", "--output-path", type=str, required=True,
        help="The path to save checkpoint files.")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(config["device"], local_rank)
        if config["device"] == "cuda":
            torch.cuda.set_device(local_rank)

        torch.distributed.init_process_group(backend=config["ddp_backend"])
    else:
        device = torch.device(config["device"])

    pipeline = dwm.common.create_instance_from_config(
        config["pipeline"], output_path=None, config=config, device=device)
    print("The pipeline is loaded.")

    os.makedirs(args.output_path, exist_ok=True)
    for i_id, i in enumerate(config["inputs"]):
        i["batch"] = {
            k: torch.tensor(v) if k != "clip_text" else v
            for k, v in i["batch"].items()
        }
        with torch.no_grad():
            if (
                "sequence_length_per_iteration" in
                config["pipeline"]["inference_config"]
            ):
                latent_shape = tuple(
                    i["latent_shape"][:1] + [
                        config["pipeline"]["inference_config"]
                        ["sequence_length_per_iteration"],
                    ] + i["latent_shape"][2:]
                )
                pipeline_output = pipeline.autoregressive_inference_pipeline(
                    **{
                        k: latent_shape if k == "latent_shape" else v
                        for k, v in i.items()
                    })
            else:
                pipeline_output = pipeline.inference_pipeline(**i)

            output_images = pipeline_output["images"]
            collected_images = [
                output_images.cpu().unflatten(0, i["latent_shape"][:3])
            ]

        stacked_images = torch.stack(collected_images)
        resized_images = torch.nn.functional.interpolate(
            stacked_images.flatten(0, 3),
            tuple(pipeline.inference_config["preview_image_size"][::-1])
        )
        resized_images = resized_images.view(
            *stacked_images.shape[:4], -1, *resized_images.shape[-2:])

        if not ddp or torch.distributed.get_rank() == 0:
            if i["latent_shape"][1] == 1:
                # [C, B * T * S * H, V * W]
                preview_tensor = resized_images.permute(4, 1, 2, 0, 5, 3, 6)\
                    .flatten(-2).flatten(1, 4)
                image_output_path = os.path.join(
                    args.output_path, "{}.png".format(i_id))
                torchvision.transforms.functional.to_pil_image(preview_tensor)\
                    .save(image_output_path)
            else:
                # [T, C, B * S * H, V * W]
                preview_tensor = resized_images.permute(2, 4, 1, 0, 5, 3, 6)\
                    .flatten(-2).flatten(2, 4)
                video_output_path = os.path.join(
                    args.output_path, "{}.mp4".format(i_id))
                dwm.utils.preview.save_tensor_to_video(
                    video_output_path, "libx264", i["batch"]["fps"][0].item(),
                    preview_tensor)

            print("{} done".format(i_id))
