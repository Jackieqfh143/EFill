export TORCH_HOME=$(pwd)  && export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file ./acc_config.yaml ./run.py --configs ./config/celeba-hq_local.yaml