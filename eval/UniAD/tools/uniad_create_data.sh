
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/create_data.py --dataset nuscenes --root-path UniAD/data/nuscenes \
       --out-dir ./data/infos \
       --extra-tag nuscenes \
       --version v1.0-collision \
       --canbus ../../nuscenes \ 

       
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python tools/create_data.py --dataset nuscenes --root-path ./data/nuscenes \
#        --out-dir ./data/infos \
#        --extra-tag nuscenes \
#        --version v1.0-mini \
#        --canbus ./data/nuscenes \