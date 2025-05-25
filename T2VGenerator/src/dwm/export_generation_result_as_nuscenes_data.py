import sys
# sys.path.append('/home/zhoujiawei/OpenDWM/src')
# sys.path.append('/home/zhoujiawei/OpenDWM/externals/TATS/tats/fvd')
import argparse
import dwm.common
import json
import numpy as np
import os
import torch
import diffusers
import torchvision
from PIL import Image
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '12382'
# os.environ['RANK'] = '0'
# os.environ['WORLD_SIZE'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'

def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to run the diffusion model to generate data for"
        "detection evaluation.")
    parser.add_argument(
        "-c", "--config-path", type=str, default='/home/zhoujiawei/OpenDWM/configs/unimlvg/generate_result_as_nuscenes_data.json',
        help="The config to load the train model and dataset.")
    parser.add_argument(
        "-o", "--output-path", type=str, default='/home/zhoujiawei/OpenDWM/generation_result',
        help="The path to save checkpoint files.")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    inference_config= config["pipeline"]['inference_config']
    # set distributed training (if enabled), log, random number generator, and
    # load the checkpoint (if required).
    ddp = "LOCAL_RANK" in os.environ
    # ddp=False
    if ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(config["device"], local_rank)
        if config["device"] == "cuda":
            torch.cuda.set_device(local_rank)

        torch.distributed.init_process_group(backend=config["ddp_backend"])
    else:
        device = torch.device(config["device"])

    # setup the global state
    if "global_state" in config:
        for key, value in config["global_state"].items():
            dwm.common.global_state[key] = \
                dwm.common.create_instance_from_config(value)

    should_log = (ddp and local_rank == 0) or not ddp
    # should_save = not torch.distributed.is_initialized() or \
    #     torch.distributed.get_rank() == 0
    

    # vae=diffusers.AutoencoderKL.from_pretrained(
    #         "/data/Diffusion_models/stable-diffusion-3-medium-diffusers", subfolder="vae")
    should_log = (ddp and local_rank == 0) or not ddp

    pipeline = dwm.common.create_instance_from_config(
        config["pipeline"], output_path=args.output_path, config=config,
        device=device)
    if should_log:
        print("The pipeline is loaded.")

    # load the dataset
    validation_dataset = dwm.common.create_instance_from_config(
        config["validation_dataset"])
    vae=pipeline.vae
    if ddp:
        validation_datasampler = \
            torch.utils.data.distributed.DistributedSampler(
                validation_dataset)
        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset,
            **dwm.common.instantiate_config(config["validation_dataloader"]),
            sampler=validation_datasampler)
    else:
        validation_datasampler = None
        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset,
            **dwm.common.instantiate_config(config["validation_dataloader"]))

    if should_log:
        print("The validation dataset is loaded with {} items.".format(
            len(validation_dataset)))

    if ddp:
        validation_datasampler.set_epoch(0)

    for batch in validation_dataloader:
        batch_size, sequence_length, view_count = batch["vae_images"].shape[:3]
        origin_dir="../nuscenes"
        origin_paths=[
            os.path.join(origin_dir, k["filename"])
            for i in batch["sample_data"]
            for j in i[1:]
            for k in j if not k["filename"].endswith(".bin")
        ]
        first_6_origin_paths=[
            os.path.join(origin_dir, k["filename"])
            for i in batch["sample_data"]
            for j in i[:1]
            for k in j if not k["filename"].endswith(".bin")
        ]
        # print(first_6_origin_paths)
        # print(torch.tensor(batch['sample_data']).shape)
        latent_height = batch["vae_images"].shape[-2] // \
            (2 ** (len(vae.config.down_block_types) - 1))
        latent_width = batch["vae_images"].shape[-1] // \
            (2 ** (len(vae.config.down_block_types) - 1))
        if "sequence_length_per_iteration" in inference_config:
            latent_shape = (
                batch_size,
                inference_config["sequence_length_per_iteration"],
                view_count, vae.config.latent_channels, latent_height,
                latent_width
            )
            with torch.no_grad():
                pipeline_output = pipeline.autoregressive_inference_pipeline(
                    latent_shape, batch, "pt")
        else:
            latent_shape = (
                batch_size, sequence_length, view_count,
                vae.config.latent_channels, latent_height,
                latent_width
            )
            with torch.no_grad():
                pipeline_output = pipeline.inference_pipeline(latent_shape,batch, "pil")

        if "images" in pipeline_output:
            
            
            paths = [
                os.path.join(args.output_path, k["filename"])
                for i in batch["sample_data"]
                for j in i[1:]
                for k in j if not k["filename"].endswith(".bin")
            ]
            first_6_output_path=[
                os.path.join(args.output_path, k["filename"])
                for i in batch["sample_data"]
                for j in i[:1]
                for k in j if not k["filename"].endswith(".bin")
            ]
            image_results = pipeline_output["images"]  # Shape [222, 3, 256, 448]
            image_sizes = batch["image_size"].flatten(0, 2)[(6*batch_size):]
            print(len(paths))
            print(len(image_results))
            print(len(image_sizes))
            #插入
            for path, image, image_size in zip(paths, image_results, image_sizes):
                dir = os.path.dirname(path)
                os.makedirs(dir, exist_ok=True)
                
                # 处理单个图像，不是整个批次
                if isinstance(image, torch.Tensor):
                    # 确保是单个图像的张量，形状应该是[3, H, W]
                    if image.dim() == 3 and image.shape[0] == 3:  # CHW格式
                        # 转换为PIL图像 (对单个图像使用ToPILImage)
                        image_pil = torchvision.transforms.ToPILImage()(image.cpu())
                    else:
                        raise ValueError(f"Unexpected image tensor shape: {image.shape}")
                else:
                    # 假设image已经是PIL图像
                    image_pil = image
                    
                # 使用PIL的resize方法调整大小
                resized_image = image_pil.resize(tuple(image_size.int().tolist()), Image.BILINEAR)
                # 保存图像
                resized_image.save(path, quality=95)
            # first_origin_path=origin_paths[0:6]
            # first_output_path=paths[0:6]
            for sensor_origin_path, sensor_output_path in zip(first_6_origin_paths, first_6_output_path):
                # 文件复制操作 - 从原始路径复制到输出路径并强制替换
                if os.path.exists(sensor_origin_path):
                    print(f"Copying file from {sensor_origin_path} to {sensor_output_path}")
                    try:
                        # 确保目标目录存在
                        output_dir = os.path.dirname(sensor_output_path)
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # 从原始路径复制到输出路径，强制替换已存在的文件
                        import shutil
                        shutil.copy2(sensor_origin_path, sensor_output_path)
                        print(f"Successfully copied file to {sensor_output_path}")
                    except Exception as e:
                        print(f"Error copying file: {e}")
                else:
                    print(f"Skipping file copy: Source file does not exist at {sensor_origin_path}")
            
# 添加复制.bin文件的逻辑
            import shutil
            
            # 获取所有.bin文件的路径
            bin_origin_paths = [
                os.path.join(origin_dir, k["filename"])
                for i in batch["sample_data"]
                for j in i
                for k in j if k["filename"].endswith(".bin")
            ]
            
            # 获取所有.bin文件的目标路径
            bin_output_paths = [
                os.path.join(args.output_path, k["filename"])
                for i in batch["sample_data"]
                for j in i
                for k in j if k["filename"].endswith(".bin")
            ]
            
            print(f"Found {len(bin_origin_paths)} .bin files to copy")
            
            # 复制所有.bin文件
            for bin_origin_path, bin_output_path in zip(bin_origin_paths, bin_output_paths):
                if os.path.exists(bin_origin_path):
                    # print(f"Copying bin file from {bin_origin_path} to {bin_output_path}")
                    try:
                        # 确保目标目录存在
                        output_dir = os.path.dirname(bin_output_path)
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # 从原始路径复制到输出路径，强制替换已存在的文件
                        shutil.copy2(bin_origin_path, bin_output_path)
                        # print(f"Successfully copied bin file to {bin_output_path}")
                    except Exception as e:
                        print(f"Error copying bin file: {e}")
                else:
                    print(f"Skipping bin file copy: Source file does not exist at {bin_origin_path}")
        