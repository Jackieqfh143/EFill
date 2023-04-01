from data.dataset import Dataset
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from evaluate.evaluation import validate
from prefetch_generator import BackgroundGenerator
from modules.trainer import InpaintingModel
from complexity import print_network_params
from tqdm import tqdm
import random
import numpy as np
import warnings
import yaml
import argparse
import torch
import time
import os

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def set_random_seed(random_seed=666,deterministic=False):
    if random_seed is not None:
        print("Set random seed as {}".format(random_seed))
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        if deterministic:
            warnings.warn('You have chosen to seed training. '
                          'This will turn on the CUDNN deterministic setting, '
                          'which can slow down your training considerably! '
                          'You may see unexpected behavior when restarting '
                          'from checkpoints.')
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            #for faster training
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

parse = argparse.ArgumentParser()
parse.add_argument('--configs',type=str,dest='configs',default="./config/configs.yaml",help='path to the config file')
arg = parse.parse_args()

with open(arg.configs, 'r') as f:
    opt = OmegaConf.create(yaml.safe_load(f))

os.environ["CUDA_VISIBLE_DEVICES"]=','.join([str(idx) for idx in opt.gpuIDs])
seed = opt.seed
set_random_seed(seed)

inpaintingModel = InpaintingModel(opt=opt)
accelerator = inpaintingModel.accelerator




#save configs
opt.saveName = inpaintingModel.saveName
opt.iteration = inpaintingModel.iteration
with open(inpaintingModel.saveDir+'/configs.yaml','w') as fp:
    OmegaConf.save(config=opt,f=fp.name)


train_dataset = Dataset(opt.train_dataDir, opt.maskDir, opt.maskType, opt.targetSize,augment=True,
                                training=True,center_crop = opt.center_crop,get_edge=True,
                        edge_type= opt.edge_type)

val_dataset = Dataset(opt.val_dataDir, opt.val_maskDir, opt.val_maskType, opt.targetSize,center_crop = opt.center_crop,
                                training=False,get_edge=True,edge_type= opt.edge_type)
# get dataloader
train_dataloader = DataLoaderX(train_dataset,
                        batch_size=opt.batchSize, shuffle=True, drop_last=False,num_workers=opt.num_workers,
                        pin_memory=False)  # set in_memory= False when GPU memory is limited


val_dataloader = DataLoaderX(val_dataset,
                        batch_size=opt.val_batchSize, shuffle=False, drop_last=False,num_workers=opt.num_workers,
                        pin_memory=False)

#prepare models for accelerator
train_dataloader,*inpaintingModel.acc_args = accelerator.prepare(train_dataloader,*inpaintingModel.acc_args)

training_magnitude = len(train_dataloader)
start_time = time.time()
count = int(inpaintingModel.iteration)
inpaintingModel.count = count
val_loss_mean = 1e3
best_val_loss = 1e3
val_losses = []
metric_str_list = []
val_count = 0
best_metric_message = ''
cur_metric_messgae = ''
ema_metric_message = ''
epoch = int(inpaintingModel.iteration) // training_magnitude
min_lr_steps = min(1,int(0.5 * training_magnitude / opt.val_step))
lr_steps = opt.lr_steps if opt.lr_steps * opt.val_step >= training_magnitude else min_lr_steps

opt.lr_steps = lr_steps
accelerator.print('*********training congfigs*********')
accelerator.print(OmegaConf.to_yaml(opt))
accelerator.print('***********************************')


if opt.debug:
    accelerator.print(inpaintingModel.G_Net)
print_network_params(inpaintingModel.G_Net,opt.Generator)

accelerator.print(f'From epoch {epoch}...')

def validation(val_type='default',enable_ema=False):
    global cur_metric_messgae,ema_metric_message
    accelerator.print(f"-------{val_type} Validate--------")
    inpaintingModel.eval()
    loss_mean_val = 0.0
    # save checkpoints for validate
    accelerator.wait_for_everyone()
    accelerator.print('stopping process...')
    accelerator.print(f'saving checkpoints for {val_type} testing...')
    save_path = os.path.join(inpaintingModel.saveDir, f"last_G_{val_type}.pth")

    if enable_ema:
        inpaintingModel.ema_G.store(inpaintingModel.G_Net.parameters())
        inpaintingModel.ema_G.copy_to(inpaintingModel.G_Net.parameters())

    if accelerator.is_main_process:
        unwrap_model = accelerator.unwrap_model(inpaintingModel.G_Net)
        accelerator.save(unwrap_model.state_dict(), save_path)

    accelerator.print(f'Loading checkpoints for {val_type} testing...')
    inpaintingModel.load_network(save_path, inpaintingModel.G_Net)
    accelerator.wait_for_everyone()
    # only validate once by main process in distributed training
    # use one single gpu for validate
    if accelerator.is_main_process:
        all_val_real_ims = []
        all_val_fake_ims = []
        all_val_masked_ims = []
        val_count = 0
        for batch in tqdm(val_dataloader):
            for i, im in enumerate(batch):
                im = im.to(inpaintingModel.device)
                batch[i] = im
            val_real_ims, val_fake_ims, val_masked_ims = inpaintingModel.validate(batch, val_count)

            all_val_real_ims += val_real_ims
            all_val_fake_ims += val_fake_ims
            all_val_masked_ims += val_masked_ims
            val_count += 1
            if val_count % opt.max_val_batches == 0:
                break

        # save validate results
        inpaintingModel.save_results(all_val_real_ims, all_val_fake_ims, all_val_masked_ims)
        val_save_dir = os.path.join(inpaintingModel.val_saveDir, 'val_results')
        val_loss_dict, loss_mean_val, metric_messgae = validate(real_imgs_dir=val_save_dir,
                                                                    comp_imgs_dir=val_save_dir,
                                                                    device=inpaintingModel.device,
                                                                    get_FID=True, get_LPIPS=True,
                                                                    get_IDS=False)

        if enable_ema:
            ema_metric_message = metric_messgae.replace('Current','Current EMA')
            metric_str_list.append(ema_metric_message)
        else:
            cur_metric_messgae = metric_messgae
            metric_str_list.append(cur_metric_messgae)
        val_losses.append(loss_mean_val)

        if len(val_losses) > 0:
            global best_val_loss,best_metric_message
            best_val_loss = min(val_losses)
            best_val_loss_idx = val_losses.index(min(val_losses))
            best_metric_message = metric_str_list[best_val_loss_idx].replace('Current', 'Best')

        inpaintingModel.recorder.add_scalar('loss_mean', loss_mean_val, val_count)

    # save checkpoints
    accelerator.wait_for_everyone()
    accelerator.print('stopping process...')
    accelerator.print('saving the final checkpoints...')
    if accelerator.is_main_process:
        inpaintingModel.save_network(loss_mean_val,val_type=val_type)


def decrease_lr(lr_decrease = False):
    if lr_decrease:
        print(f'Decreasing learning rate for process {accelerator.process_index}...')
        inpaintingModel.adjust_learning_rate(inpaintingModel.current_lr, opt.min_lr,
                                             inpaintingModel.G_opt, epoch, lr_factor=opt.lr_factor,
                                             warm_up=False, name='lr')

        if opt.gan_loss:
            inpaintingModel.adjust_learning_rate(inpaintingModel.current_d_lr, opt.min_d_lr,
                                             inpaintingModel.D_opt, epoch, lr_factor=opt.lr_factor,
                                             warm_up=False, name='d_lr')


while count < opt.max_iter:

    for batch in tqdm(train_dataloader):
        if count % opt.val_step != 0:
            inpaintingModel.forward(batch=batch, count=count)
            inpaintingModel.optimize_params()

            if count % opt.print_loss_step == 0:
                inpaintingModel.reduce_loss()
                time_cost = time.time() - start_time
                # accelerator.print(inpaintingModel.lossDict)
                loss_str = 'Iteration: {:d}/{:d}|| lr:{:.5f} time_taken:{:.2f} s '.format(
                    count, opt.max_iter, inpaintingModel.current_lr, time_cost)
                for k, v in inpaintingModel.print_loss_dict.items():
                    loss_str += " {}: {:.2f}".format(k, v)

                inpaintingModel.print_loss_dict.clear()

                if best_metric_message != '':
                    accelerator.print('\n' + best_metric_message + '\n')
                accelerator.print(loss_str)
                if cur_metric_messgae != '':
                    accelerator.print(cur_metric_messgae + '\n')
                    if opt.enable_ema:
                        accelerator.print(ema_metric_message + '\n')

                start_time = time.time()


        elif count != 0 or opt.debug:

            validation(val_type='default')

            if opt.enable_ema:
                validation(val_type='ema', enable_ema=True)

                if not val_losses[-1] < val_losses[-2]:
                # do not restore the paramter of G_Net if ema_G performs better
                    inpaintingModel.ema_G.restore(inpaintingModel.G_Net.parameters())

            if accelerator.is_main_process:

                opt.lr = round(inpaintingModel.current_lr, 5)

                opt.d_lr = round(inpaintingModel.current_d_lr, 6)

                opt.iteration = count

                inpaintingModel.iteration = count

                # update performance record

                with open(inpaintingModel.saveDir + '/performance.txt', 'a') as fp:

                    performance_str = f'Iter: {count} ' + f' current_lr: {round(inpaintingModel.current_lr, 6)} ' + cur_metric_messgae.replace(

                        'Current', ' ') + '\n'

                    fp.write(performance_str)

                    if opt.enable_ema:
                        performance_str = f'Iter: {count} ' + f' current_lr: {round(inpaintingModel.current_lr, 6)} ' + ema_metric_message.replace(

                            'Current', ' ') + '\n'

                        fp.write(performance_str)

                lr_decrease = False

                if opt.mode == 1:

                    if len(val_losses) >= lr_steps:

                        if sum(val_losses[-lr_steps:]) / lr_steps > val_loss_mean:

                            # if val_losses[-1] > val_loss_mean:

                            lr_decrease = True

                        else:

                            val_loss_mean = round(sum(val_losses[-lr_steps:]) / lr_steps, 3)

                    else:

                        val_loss_mean = round(sum(val_losses) / len(val_losses), 3)

                opt.lr_decrease = lr_decrease

                with open(inpaintingModel.saveDir + '/configs.yaml', 'w') as fp:

                    OmegaConf.save(config=opt, f=fp.name)

            accelerator.wait_for_everyone()

            with open(inpaintingModel.saveDir + '/configs.yaml', 'r') as f:

                opt = OmegaConf.create(yaml.safe_load(f))

            inpaintingModel.current_lr = opt.lr

            inpaintingModel.current_d_lr = opt.d_lr

            inpaintingModel.iteration = opt.iteration

            # Adjust lr for all process

            decrease_lr(opt.lr_decrease)

            # turn back to the training state

            accelerator.print("-------Training--------")

            inpaintingModel.train()

        count += 1

    epoch += 1
