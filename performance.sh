export PYTHONPATH=$(pwd)
python performance.py \
--dataset_name celeba \
--config_path ./config/celeba_train.yaml \
--model_path ./checkpoints/celeba_best.pth \
--mask_type thick_256 \
--target_size 256 \
--total_num 10000 \
--img_dir ./dataset/validation/Celeba/thick_256/imgs \
--mask_dir ./dataset/validation/Celeba/thick_256/masks \
--save_dir ./results