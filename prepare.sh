export PYTHONPATH=$(pwd)
python prepare_masks.py \
--dataset_name "Celeba" \
--mask_type "thick_256" \
--target_size 256 \
--aspect_ratio_kept \
--fixed_size \
--total_num  10000 \
--img_dir "/home/codeoops/CV/data/celeba/test" \
--save_dir "./dataset/validation"