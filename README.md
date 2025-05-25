# SafeMVDrive: Multi-view Safety-Critical Driving Video Synthesis in the Real World Domain

This repository contains the code for the following work:
> Challenger: SafeMVDrive: Multi-view Safety-Critical Driving Video Synthesis in the Real World Domain
> Authors: Jiawei Zhou, Linye Lyu, Zhuotao Tian, Cheng Zhuo, Yu Li

<br>
<div align="center">
  <img src="https://img.shields.io/github/license/zhoujiawei3/SafeMVDrive" alt="License">
  <a href="https://arxiv.org/abs/<ARXIV PAPER ID>"><img alt='arXiv' src="https://img.shields.io/badge/<ARXIV PAPER ID>"></a>
  <a href="https://huggingface.co/datasets/JiaweiZhou/SafeMVDrive"><img alt='Dataset' src="https://img.shields.io/badge/Dataset-SafeMVDrive-blue"></a>
  <a href="https://zhoujiawei3.github.io/SafeMVDrive/"><img alt='Project Page' src="https://img.shields.io/badge/Webpage-SafeMVDrive-green"></a>
</div>
<br>


<!-- <p align="center">
  <div align="center">Adversarial vehicle suddenly cuts in; ego vehicle slightly steers right to avoid.</div>
  <video src="assets/adv_gif/01-ezgif.com-video-to-gif-converter.gif" width="100%" style="max-width: 100%; height: auto;" autoplay loop muted playsinline></video>

  <div align="center">Rear adversarial vehicle suddenly accelerates; ego vehicle also speeds up to evade.</div>
  <video src="assets/adv_gif/02-ezgif.com-video-to-gif-converter.gif" width="100%" style="max-width: 100%; height: auto;" autoplay loop muted playsinline></video>

  <div align="center">Rear adversarial vehicle suddenly accelerates; ego vehicle changes lane left to evade.</div>
  <video src="assets/adv_gif/03-ezgif.com-video-to-gif-converter.gif" width="100%" style="max-width: 100%; height: auto;" autoplay loop muted playsinline></video>

  <div align="center">Front adversarial vehicle suddenly slows down; ego vehicle changes lane and decelerates to avoid.</div>
  <video src="assets/adv_gif/04-ezgif.com-video-to-gif-converter.gif" width="100%" style="max-width: 100%; height: auto;" autoplay loop muted playsinline></video>
</p> -->


<p align="center">
  <div align="center">Adversarial vehicle suddenly cuts in; ego vehicle slightly steers right to avoid.</div>
  <img src="assets/adv_gif/01-ezgif.com-video-to-gif-converter.gif" width="100%" style="max-width: 100%; height: auto;" />
  <div align="center">Rear adversarial vehicle suddenly accelerates; ego vehicle also speeds up to evade.</div>
  <img src="assets/adv_gif/02-ezgif.com-video-to-gif-converter.gif" width="100%" style="max-width: 100%; height: auto;" />
  <div align="center">Rear adversarial vehicle suddenly accelerates; ego vehicle changes lane left to evade.</div>
  <img src="assets/adv_gif/03-ezgif.com-video-to-gif-converter.gif" width="100%" style="max-width: 100%; height: auto;" />
  <div align="center">Front adversarial vehicle suddenly slows down; ego vehicle changes lane and decelerates to avoid.</div>
  <img src="assets/adv_gif/04-ezgif.com-video-to-gif-converter.gif" width="100%" style="max-width: 100%; height: auto;" />
</p>

## Abstract

Safety-critical scenarios are rare yet pivotal for evaluating and enhancing the robustness of autonomous driving systems. While existing methods generate safety-critical driving trajectories, simulations, or single-view videos, they fall short of meeting the demands of advanced end-to-end autonomous systems (E2E AD), which require real-world, multi-view video data. To bridge this gap, we introduce SafeMVDrive, the first framework designed to generate high-quality, safety-critical, multi-view driving videos grounded in real-world domains. SafeMVDrive strategically integrates a safety-critical trajectory generator with an advanced multi-view video generator. To tackle the challenges inherent in this integration, we first enhance scene understanding ability of the trajectory generator by incorporating visual context -- which is previously unavailable to such generator -- and leveraging a GRPO-finetuned vision-language model to achieve more realistic and context-aware trajectory generation. Second, recognizing that existing multi-view video generators struggle to render realistic collision events, we introduce a two-stage, controllable trajectory generation mechanism that produces collision-evasion trajectories, ensuring both video quality and safety-critical fidelity. Finally, we employ a diffusion-based multi-view video generator to synthesize high-quality safety-critical driving videos from the generated trajectories. Experiments conducted on an E2E AD planner demonstrate a significant increase in collision rate when tested with our generated data, validating the effectiveness of SafeMVDrive in stress-testing planning modules.

## Getting Started
 The codebase is organized into two primary modules:
- `vlm-selector/`: VLM-based Adversarial Vehicle Selector
- `two-stage-simulator/`: Two-stage Evasion Trajectory Generator
- `T2VGenerator/`: Trajectory-to-Video Generator
- `eval/`: E2E driving module evaluation

Each module requires a separate environment.

### 1. Parsing the nuScenes Dataset and model weights

#### Model Weights

First, prepare the weights for our three components.

**vlm-selector**: Get our GRPO-finetuned Qwen2.5 VL-Instruct-7B model from: [https://huggingface.co/JiaweiZhou/SafeMVDrive/tree/main/vlm-selector/checkpoint-2600](https://huggingface.co/JiaweiZhou/SafeMVDrive/tree/main/vlm-selector/checkpoint-2600).

**two-stage-simulator**: Get our diffusion-based trajectory generation model trained with one-frame context from from: [https://huggingface.co/JiaweiZhou/SafeMVDrive/tree/main/two-stage-simulator](https://huggingface.co/JiaweiZhou/SafeMVDrive/tree/main/two-stage-simulator).

**T2VGenerator**: Pretrained weights required by [UniMLVG](https://huggingface.co/wzhgba/opendwm-models/resolve/main/ctsd_unimlvg_tirda_bm_nwa_60k.pth?download=true)


Organize the directory structure as follows:

```bash
    ${CODE_ROOT}/
    ├── T2VGenerator
    ├── ...
    ├── weights
    │   ├── vlm-selector
    │   │   ├── checkpoint-2600
    │   │   │   ├── global_step2600
    │   │   │   ├── ...        
    │   ├── two-stage-simulator
    │   │   ├── config.json 
    │   │   ├── iter80000.ckpt
    │   ├── T2VGenerator
    │   │   ├── ctsd_unimlvg_tirda_bm_nwa_60k.pth
```

#### Dataset

Download the original nuScenes dataset from [nuScenes](https://www.nuscenes.org/nuscenes) and organize the directory structure as follows:

```bash
    ${CODE_ROOT}/
    ├── T2VGenerator
    ├── ...
    ├── nuscenes
    │   ├── v1.0-trainval-zip
    │   │   ├── nuScenes-map-expansion-v1.3.zip
    │   ├── can_bus
    │   ├── maps
    │   ├── samples
    │   ├── sweeps
    │   ├── v1.0-trainval

```

### 2. VLM-based Adversarial Vehicle Select

Test on CUDA 12.4.

#### Setup
```bash
conda create -n safemvdrive-vlm python=3.10
conda activate safemvdrive-vlm
cd vlm-selector/
bash setup.sh
```

#### Preprocessing and VLM Inference
```bash
cd vlm-selector/src
bash VLM_selector.sh
```
You can modify the `DATA_COUNT` in the script to change the number of samples randomly selected from the nuscenes val dataset.

### 3. Two-stage Evasion Trajectory Generation

#### Setup

Test on CUDA 11.3.

```bash
conda create -n safemvdrive-trajectory python=3.9
conda activate safemvdrive-trajectory
cd two-stage-simulator/
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113
pip install pip==24.0 # 
pip install numpy==1.23.4 # ignore confict
pip install -e .
```

Install a customized version of `trajdata`
```bash
git clone https://github.com/AIasd/trajdata.git
cd trajdata
pip install -r trajdata_requirements.txt
pip install -e .
```

Install `Pplan`
```bash
git clone https://github.com/NVlabs/spline-planner.git Pplan
cd Pplan
pip install -e .
```

#### Evasion Trajectory Generation

```bash
cd two-stage-simulator
bash two-stage-simulate.sh
```


### 4. Trajectory-to-Video Generation

#### Setup

Software requirement:
* git (>= 2.25)

```
conda create -n safemvdrive-video python=3.9
python -m pip install torch==2.5.1 torchvision==0.20.1
cd T2VGenerator
git submodule update --init --recursive
python -m pip install -r requirements.txt
```



#### Trajectory-to-Video Generation
```bash
cd T2VGenerator
bash T2VGeneration.sh
```

### 5. Evaluating End-to-End Autonomous Driving Models on the Generated Adversarial Dataset

We take [UniAD](https://github.com/OpenDriveLab/UniAD/) as an example. To evaluate it on the generated adversarial dataset (or our [SafeMVDrive](https://huggingface.co/datasets/JiaweiZhou/SafeMVDrive) dataset), follow these steps:

1. Setup environment required by UniAD and download pretrained weight.
2. Comment out the following lines (and fix indentation) in `/path/to/uniad/env/site-packages/nuscenes/eval/common/loaders.py`
```python3
if scene record['name'] in splits[eval_split]:
```

```python3
else:
    raise ValueError('Error: Requested split {} which this function cannot map to the correct NuScenes version.'
                         .format(eval_split))
```

<!-- 3. Comment out [one bug line](https://github.com/OpenDriveLab/UniAD/blob/dd161b220c22de1c874c3fb8d55979054b24d716/projects/mmdet3d_plugin/datasets/pipelines/loading.py#L48)

4. Change the code [here]https://github.com/OpenDriveLab/UniAD/blob/dd161b220c22de1c874c3fb8d55979054b24d716/projects/mmdet3d_plugin/datasets/nuscenes_e2e_dataset.py#L350:
```python3
 #mask = info['num_lidar_pts'] > 0
 mask = info['num_lidar_pts']!= -1
```
Change the code [here]https://github.com/OpenDriveLab/UniAD/blob/dd161b220c22de1c874c3fb8d55979054b24d716/projects/mmdet3d_plugin/datasets/nuscenes_e2e_dataset.py#L1131-L1134:
```python3
#eval_set_map = {
#            'v1.0-mini': 'mini_val',
#            'v1.0-trainval': 'val',
#        }
eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
            'v1.0-collision': 'collision'
        }
``` -->

3. The remaining required modifications for UniAD have been placed in `eval/UniAD`, intended to replace the corresponding files under `{ROOT_OF_UNIAD}`.

<!-- 3. Make the following modification to `{ROOT_OF_UNIAD}/mmdetection3d/mmdet3d/datasets/nuscenes_dataset.py`:
```python3
# data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
data_infos = list(sorted(data["infos"], key=lambda e: (e["scene_token"], e["timestamp"])))
``` -->
4. Create a symbolic link to the generated dataset directory at `{ROOT_OF_UNIAD}/data/nuscenes` and execute `{ROOT_OF_UNIAD}/tools/uniad_create_data.sh` to extract metadata.sh to extract metadata.
5. Run `{ROOT_OF_UNIAD}/tools/uniad_dist_eval.sh` to evaluate.

By default, we use the output `obj_box_col` as the basis for calculating the **sample level collision rate**.
To compute the **scene-level collision rate** instead, please modify the following lines accordingly:
[`planning_metrics.py` (Lines 176–179)](https://github.com/zhoujiawei3/SafeMVDrive/blob/5991bab339ce2fd26f384686c11fa3bc6c7be6a9/eval/UniAD/projects/mmdet3d_plugin/uniad/dense_heads/planning_head_plugin/planning_metrics.py#L176-L179)






