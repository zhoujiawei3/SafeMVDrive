#!/bin/bash

# 默认参数
USE_APPROXIMATE_CAN_BUS=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --use-approximate-can-bus)
            USE_APPROXIMATE_CAN_BUS="--use-approximate-can-bus"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 打印信息
if [ -n "$USE_APPROXIMATE_CAN_BUS" ]; then
    echo "使用估算 CAN BUS 信息"
else
    echo "使用真实 CAN BUS 信息"
fi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/create_data.py --dataset nuscenes --root-path UniAD/data/nuscenes \
       --out-dir ./data/infos \
       --extra-tag nuscenes \
       --version v1.0-collision \
       --canbus UniAD/data/nuscenes \
       $USE_APPROXIMATE_CAN_BUS

# 备选命令（带 v1.0-trainval）：
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python tools/create_data.py --dataset nuscenes --root-path ./data/nuscenes \
#        --out-dir ./data/infos \
#        --extra-tag nuscenes \
#        --version v1.0 \
#        --canbus ./data/nuscenes \
#        $USE_APPROXIMATE_CAN_BUS