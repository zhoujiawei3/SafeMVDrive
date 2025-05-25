# Configurations

The configuration files are in the JSON format. They include settings for the models, datasets, pipelines, or any arguments for the program.

The configs in this folder are mainly about the pipelines and consumed by the `src/dwm/train.py`. So they are named in the format of `{pipeline_name}_{model_config}_{condition_config}_{data_config}.json`.

* Pipeline name: the python script name in the `src/dwm/pipelines`.
* Model config: the most discriminative model arguments, such as `image`, `lidar`, `joint` for the holodrive models, or `spatial`, `crossview`, `temporal` for the SD models.
* Condition config: the additional input for the model, such as `ts` for the "text description per scene", `ti` for the "text description per image", `b` for the box condition, `m` for the map condition.
* Data config: `mini` for the debug purpose. Or combination of `nuscenes`, `argoverse`, `waymo`, `opendv`, for the data components. For some dataset, use `k` for "key frames", `a` for "all frames".