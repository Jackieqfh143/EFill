# Requirement

- Python  3.7
- torch   1.12.1
- accelerate  0.17.1

# Quick Start

## Set up

```shell
conda create -n efill python=3.7 
conda activate efill 
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt  
```

## Pretrained Models

[Place](https://drive.google.com/file/d/1snNOR78h8sS9gHYGM55knUhqcsJgQJ86/view?usp=share_link)  [Celeba](https://drive.google.com/file/d/164xO5TntSMXxSmOeAeNdGYI1bYbqGUj5/view?usp=share_link) 

Download the pretrained models above, and put them under the folder of ```checkpoints``` 

## Inference

```shell
cd demo 
python demo.py \
--port 8000 \
--model_path ../checkpoints/place_best.pth
```

Then, click on the link that pops up below. For example 

http://127.0.0.1:8000



# Training

> Download the dataset 

Please refer to this link [lama](https://github.com/advimman/lama) for download the dataset of CelebA-HQ and Places.

> Download the pretrained models

1. download [model](https://drive.google.com/file/d/16Zy70lIuOidQ39_m73FNnOEiV_dWlXjU/view?usp=share_link) for calculating the perceptual loss 

2. download the models [AlexNet](https://download.pytorch.org/models/alexnet-owt-7be5be79.pth) and [Inception](https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth) for metric calculation.

   ```shell
   mkdir -p ./hub/checkpoints
   cd ./hub/checkpoints
   wget https://download.pytorch.org/models/alexnet-owt-7be5be79.pth
   wget https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth
   ```

3. prepare images and masks for validation

   ```shell
   sh prepare.sh 
   ```

4. download the pretrained teacher models (**Recommend**)

​		[Place](https://drive.google.com/file/d/1iGq7CSaZwLh6ndKg6dPrSgGEWj8-3TNg/view?usp=share_link)   [Celeba-HQ](https://drive.google.com/file/d/1-dHy9Es1wBM5j3kaxiPj7u30YabwnF0i/view?usp=share_link)   [Celeba](https://drive.google.com/file/d/1VqabXkPNr2OostmcY9Yv2JGwTrLysOn7/view?usp=share_link)

​		Note: this is an optional choice. You can also train the teacher model from 		scratch.   

> Configure the accelerator

We use the framework [accelerate](https://github.com/huggingface/accelerate) to speed up the training. Before starting trainging, you should specify a config file for it. Run the following command in terminal.

```shell
accelerate config --config_file acc_config.yaml
```

> Training the teacher

Modify the ```example_train.yaml``` on the following items:

```yam
mode: 2  
Generator: Teacher_concat_WithAtt
...
```

Then run

```shell
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file ./acc_config.yaml ./run.py --configs ./config/example_train.yaml
```

> Training EFill 

Modify the ```example_train.yaml``` on the following items:

```yam
mode:1
Generator: DistInpaintModel_SPADE_IN_LFFC_Base_concat_WithAtt
st_TeacherPath：./checkpoints/celeba-hq_latest.pth
...
```



# Evaluate 

> Prepare the images and masks 

```shell
python prepare_masks.py \
--dataset_name "Celeba" \
--mask_type "thick_256" \
--target_size 256 \
--aspect_ratio_kept \
--fixed_size \
--total_num  10000 \
--img_dir "/home/codeoops/CV/data/celeba/test" \
--save_dir "./dataset/validation"
```

> Evaluate the performance

```shell
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
```



# Acknowledgement

Our code is built upon the following repositories:

-  LaMa https://github.com/advimman/lama
-  CrFill https://github.com/zengxianyu/crfill
-  DeepFill https://github.com/JiahuiYu/generative_inpainting

