import math
import argparse, yaml
import utils
import os
from tqdm import tqdm
import logging
import sys
import time
import importlib
import glob
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from datas.utils import create_datasets

parser = argparse.ArgumentParser(description='SPIN')
## yaml configuration files
parser.add_argument('--config', type=str, default=None, help = 'pre-config file for training')
parser.add_argument('--resume', type=str, default=None, help = 'resume training or not')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.config:
       opt = vars(args)
       yaml_args = yaml.load(open(args.config), Loader=yaml.FullLoader)
       opt.update(yaml_args)
    ## set visibel gpu   
    gpu_ids_str = str(args.gpu_ids).replace('[','').replace(']','')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_ids_str)

    ## select active gpu devices
    device = None
    if args.gpu_ids is not None and torch.cuda.is_available():
        print('use cuda & cudnn for acceleration!')
        print('the gpu id is: {}'.format(args.gpu_ids))
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        print('use cpu for training!')
        device = torch.device('cpu')
    torch.set_num_threads(args.threads)

    ## create dataset for training and validating
    train_dataloader, valid_dataloaders = create_datasets(args)

    ## definitions of model
    try:
        model = utils.import_module('models.{}'.format(args.model)).create_model(args)
    except Exception:
        raise ValueError('not supported model type! or something')
    model = nn.DataParallel(model).to(device)

    ## resume training
    start_epoch = 1
    assert args.resume is not None
    ckpt_files = glob.glob(os.path.join(args.resume, 'models', "*.pt"))
    if len(ckpt_files) != 0:
        ckpt_files = sorted(ckpt_files, key=lambda x: int(x.replace('.pt','').split('_')[-1]))
        ckpt = torch.load(ckpt_files[-1])
        prev_epoch = ckpt['epoch']
        start_epoch = prev_epoch + 1
        model.load_state_dict(ckpt['model_state_dict'])
        stat_dict = ckpt['stat_dict']
        print('select {}, resume training from epoch {}.'.format(ckpt_files[-1], start_epoch))

    ## print architecture of model
    time.sleep(3) # sleep 3 seconds 
    print(model)

    epoch = 1
    torch.set_grad_enabled(False)
    test_log = ''
    model = model.eval()
    for valid_dataloader in valid_dataloaders:
        avg_psnr, avg_ssim = 0.0, 0.0
        name = valid_dataloader['name']
        loader = valid_dataloader['dataloader']
        count = 0 
        for lr, hr in tqdm(loader, ncols=80):
            count += 1

            lr, hr = lr.to(device), hr.to(device)
            torch.cuda.empty_cache()

            sr = model(lr)
            # quantize output to [0, 255]
            hr = hr.clamp(0, 255)
            sr = sr.clamp(0, 255)

            out_img = sr.detach()[0].float().cpu().numpy()
            out_img = np.transpose(out_img, (1, 2, 0))

            output_folder = os.path.join(args.output_folder, str(name))

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            output_folder = os.path.join(output_folder, str(count) + '_x' + str(args.upscale) + '.png')
            cv2.imwrite(output_folder, out_img[:, :, [2, 1, 0]]) #

            # conver to ycbcr
            if args.colors == 3:
                hr_ycbcr = utils.rgb_to_ycbcr(hr)
                sr_ycbcr = utils.rgb_to_ycbcr(sr)
                hr = hr_ycbcr[:, 0:1, :, :]
                sr = sr_ycbcr[:, 0:1, :, :]

            hr = hr[:, :, args.upscale:-args.upscale, args.upscale:-args.upscale]
            sr = sr[:, :, args.upscale:-args.upscale, args.upscale:-args.upscale]                

            psnr = utils.calc_psnr(sr, hr)       
            ssim = utils.calc_ssim(sr, hr)         
            avg_psnr += psnr
            avg_ssim += ssim

        avg_psnr = round(avg_psnr/len(loader) + 5e-3, 2)
        avg_ssim = round(avg_ssim/len(loader) + 5e-5, 4)

        stat_dict[name]['psnrs'].append(avg_psnr)
        stat_dict[name]['ssims'].append(avg_ssim)
        if stat_dict[name]['best_psnr']['value'] < avg_psnr:
            stat_dict[name]['best_psnr']['value'] = avg_psnr
            stat_dict[name]['best_psnr']['epoch'] = epoch
        if stat_dict[name]['best_ssim']['value'] < avg_ssim:
            stat_dict[name]['best_ssim']['value'] = avg_ssim
            stat_dict[name]['best_ssim']['epoch'] = epoch
        test_log += '[{}-X{}], PSNR/SSIM: {:.2f}/{:.4f} (Best: {:.2f}/{:.4f}, Epoch: {}/{})\n'.format(
            name, args.upscale, float(avg_psnr), float(avg_ssim), 
            stat_dict[name]['best_psnr']['value'], stat_dict[name]['best_ssim']['value'], 
            stat_dict[name]['best_psnr']['epoch'], stat_dict[name]['best_ssim']['epoch'])

    print(test_log)
