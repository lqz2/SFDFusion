from modules import *
import os
import numpy as np
from utils.evaluator import Evaluator
import torch
from utils.img_read import *
import logging
from kornia.metrics import AverageMeter
from tqdm import tqdm
import warnings
import yaml
from configs import from_dict
import dataset
from torch.utils.data import DataLoader
from thop import profile, clever_format
import time
import cv2
import argparse


warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
log_f = '%(asctime)s | %(filename)s[line:%(lineno)d] | %(levelname)s | %(message)s'
logging.basicConfig(level='INFO', format=log_f)


def fuse(args):
    
    fuse_out_folder = args.out_dir
    if not os.path.exists(fuse_out_folder):
        os.makedirs(fuse_out_folder)

    fuse_net = Fuse()
    ckpt = torch.load(args.ckpt_path, map_location=device)
    fuse_net.load_state_dict(ckpt['fuse_net'])
    fuse_net.to(device)
    fuse_net.eval()

    time_list = []

    img_names = [i for i in os.listdir(args.ir_path)]
    ir_imgs=[img_read(os.path.join(args.ir_path, i),mode='L').unsqueeze(0) for i in img_names]
    vi_imgs=[img_read(os.path.join(args.vi_path, i),mode='YCbCr')[0].unsqueeze(0) for i in img_names]
    for idx,img in enumerate(ir_imgs):
        _,_, h, w = img.shape
        if h // 2 != 0 or w // 2 != 0:
            ir_imgs[idx] = ir_imgs[idx][:, : h // 2 * 2, : w // 2 * 2]
            vi_imgs[idx] = vi_imgs[idx][:, : h // 2 * 2, : w // 2 * 2]
    data_list=zip(ir_imgs, vi_imgs, img_names)
    with torch.no_grad():
        logging.info(f'fusing images ...')
        iter = tqdm(data_list, total=len(img_names), ncols=80)
        for data_ir, data_vi,img_name in iter:
            data_vi, data_ir = data_vi.to(device), data_ir.to(device)

            ts = time.time()
            fus_data, _, _ = fuse_net(data_ir, data_vi)
            # print(fus_data.shape)
            te = time.time()
            time_list.append(te - ts)
            if args.mode == 'gray':
                fi = np.squeeze((fus_data * 255).cpu().numpy()).astype(np.uint8)
                img_save(fi, img_name, fuse_out_folder)
            elif args.mode == 'RGB':
                vi_cbcr = vi_cbcr.to(device)
                fi = torch.cat((fus_data, vi_cbcr), dim=1)
                fi = ycbcr_to_rgb(fi)
                fi = tensor_to_image(fi) * 255
                fi = fi.astype(np.uint8)
                img_save(fi, img_name, fuse_out_folder, mode='RGB')

    # logging.info(f'fusing images done!')
    # logging.info(f'time: {np.round(np.mean(time_list[1:]), 6)}s')





if __name__ == "__main__":
    
    parse = argparse.ArgumentParser()
    parse.add_argument('--ckpt_path', type=str, default=f'models/model-1_1_10_1.pth')
    parse.add_argument('--ir_path', type=str, default='./ir_imgs/')
    parse.add_argument('--vi_path', type=str, default='./vi_imgs/')
    parse.add_argument('--out_dir', type=str, default=f'test_result/fuse_result/')
    parse.add_argument('--mode', type=str, default='gray')
    args = parse.parse_args()

    fuse(args)
    
