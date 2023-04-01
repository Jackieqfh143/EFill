import time
import os
import re
import glob
import torch
from accelerate import Accelerator
from utils.util import checkDir,get_file_info
import traceback

class BaseModel():
    def __init__(self,opt):
        self.opt = opt
        self.iteration = self.opt.iteration
        self.loss_names = []
        self.accelerator = Accelerator(gradient_accumulation_steps=opt.acc_steps)
        create_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        modelName = self.opt.Generator
        datasetName = self.opt.datasetName
        loss_type = self.opt.loss_type
        if self.opt.saveName != '':
            self.saveDir = '{}/{}'.format(self.opt.saveDir, self.opt.saveName)
            self.saveName = self.opt.saveName
        else:
            save_Name = '{}_{}_{}_{}'.format(modelName,datasetName, loss_type, create_time)
            self.saveDir = '{}/{}'.format(self.opt.saveDir, save_Name)
            self.saveName = save_Name

        self.log_path = '{}/{}'.format(self.opt.log_path, self.saveName)
        self.val_saveDir = '{}/{}'.format(self.opt.val_saveDir, self.saveName)
        checkDir([self.saveDir, self.log_path,self.val_saveDir])


    def load_network(self,modelDir,model,load_last=False,load_from_iter=None):
        with self.accelerator.main_process_first():
            self.accelerator.print('loading pretrained model from disk...')
            #load best model
            if os.path.isdir(modelDir):
                if load_from_iter != None:
                    model_path = self.find_model_by_iter(modelDir, load_from_iter)
                else:
                    if not load_last:
                        model_path = self.find_model(modelDir,model_name='best')
                    else:
                        model_path = self.find_model(modelDir, model_name='last')

                self.iteration = int(model_path.split('G-step=')[1].split('_')[0])
            else:
                model_path = modelDir

            self.accelerator.print('load from checkpoints: ', model_path)

            try:
                try:
                    model.load_state_dict(torch.load(model_path, map_location='cpu'))
                except Exception as e:
                    self.accelerator.print('Unmatched checkpoint! Turning off the strict mode... ')
                    model.load_state_dict(torch.load(model_path, map_location='cpu'),strict=False)
                else:
                    self.accelerator.print('loading network done.')
            except Exception as e:
                traceback.print_exc()
                self.accelerator.print('Unable to load pretrained model from disk!')
            else:
                self.accelerator.print('loading network done.')


    def find_model(self,modelDir,model_name='best'):
        checkpoints = glob.glob(os.path.join(modelDir, 'G-*.pth'))
        if model_name == 'best':
            loss_mean = lambda x: float(x.split('loss=')[1].rstrip('.pth'))
            checkpoints = sorted(checkpoints, key=loss_mean, reverse=True)  # sorted checkpoints by loss mean
            target_model = checkpoints[-1]

        elif model_name == 'last':
            iteration = lambda x: int(x.split('G-step=')[1].split('_')[0])
            checkpoints = sorted(checkpoints, key=iteration)  # sorted checkpoints by iteration
            target_model = checkpoints[-1]

        else:
            if model_name in checkpoints:
                target_model = model_name
            else:
                model_name_ = "best" if not self.opt.load_last else "last"
                self.accelerator.print(f'Failed to find model for Iteration {model_name}! Will load from the {model_name_} instead...')
                target_model = self.find_model(modelDir,model_name=model_name_)

        return target_model

    def find_model_by_iter(self,modelDir,iteration):
        checkpoints = sorted(glob.glob(os.path.join(modelDir, 'G-*.pth')))
        for model_name in checkpoints:
            temp = int(model_name.split('G-step=')[1].split('_')[0])
            if temp == iteration:
                return model_name
        else:
            model_name_ = "best" if not self.opt.load_last else "last"
            self.accelerator.print(
                f'Failed to find model for Iteration {iteration}! Will load from the {model_name_} instead...')
            target_model = self.find_model(modelDir, model_name=model_name_)

            return target_model









