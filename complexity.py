from torchscan.crawler import crawl_module
from fvcore.nn import FlopCountAnalysis
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm

def parse_shapes(input):
    if isinstance(input, list) or isinstance(input,tuple):
        out_shapes = [item.shape[1:] for item in input]
    else:
        out_shapes = input.shape[1:]

    return out_shapes

def flop_counter(model,input):
    try:
        cuda_infer_time(model,input)
        module_info = crawl_module(model, parse_shapes(input))
        flops = sum(layer["flops"] for layer in module_info["layers"])
    except Exception as e:
        print(f'\nflops counter came across error: {e} \n')
        try:
            print('try another counter...\n')
            if isinstance(input, list):
                input = tuple(input)
            flops = FlopCountAnalysis(model, input).total()
        except Exception as e:
            print(e)
            raise e
        else:
            flops = flops / 1e9
            print(f'FLOPs : {flops:.5f}')
            return flops

    else:
        flops = flops / 1e9
        print(f'FLOPs : {flops:.5f}')
        return flops

def print_network_params(model,model_name):
    num_params = 0
    if isinstance(model,list):
        for m in model:
            for param in m.parameters():
                num_params += param.numel()
        print('[Network %s] Total number of parameters : %.5f M' % (model_name, num_params / 1e6))

    else:
        for param in model.parameters():
            num_params += param.numel()
        print('[Network %s] Total number of parameters : %.5f M' % (model_name, num_params / 1e6))



def cuda_infer_time(model,input,model_name="",device="cuda"):
    print("Evaluating the inference time....\n")

    model.eval().to(device)
    if isinstance(input,list):
        for i,item in enumerate(input):
            input[i] = item.to(device)
        dummy_input = input
    else:
        dummy_input = [input.to(device)]

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings=np.zeros((repetitions,1))
    #GPU-WARM-UP
    print("GPU Warm up...\n")
    for _ in tqdm(range(10)):
        _ = model(*dummy_input)

    print("Measuring the performance...\n")
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in tqdm(range(repetitions)):
            starter.record()
            _ = model(*dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    fps = 1000 / mean_syn
    print('[Network %s] Average inference time cost : %.3f ms' % (model_name, mean_syn))
    print('[Network %s] Frame per second (FPS) : %.2f' % (model_name, fps))
    return mean_syn




if __name__ == '__main__':
    x = torch.randn(1,256,32,32)

