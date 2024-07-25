import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from modules import *
from utils.loss import *
from utils.get_params_group import get_param_groups
import kornia
from kornia.metrics import AverageMeter
from configs import *
import logging
import yaml
import dataset
from tqdm import tqdm
import argparse
import numpy as np

import wandb


def to_device(mlist, device):
    for module in mlist:
        module.to(device)


def init_params_group(mlist):
    pg0, pg1, pg2 = [], [], []
    for m in mlist:
        pg = get_param_groups(m)
        pg0.extend(pg[0])
        pg1.extend(pg[1])
        pg2.extend(pg[2])
    return pg0, pg1, pg2


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train(cfg_path, wb_key):
    config = yaml.safe_load(open(cfg_path))
    cfg = from_dict(config)
    set_seed(cfg.seed)
    log_f = '%(asctime)s | %(filename)s[line:%(lineno)d] | %(levelname)s | %(message)s'
    logging.basicConfig(level='INFO', format=log_f)
    # wandb
    wandb.login(key=wb_key)  # wandb api key
    runs = wandb.init(project=cfg.project_name, name=cfg.dataset_name + '_' + cfg.exp_name, config=cfg, mode=cfg.wandb_mode)

    # Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fuse_net = Fuse()
    module_list = [fuse_net]
    to_device(module_list, device)

    optimizer = torch.optim.Adam(fuse_net.parameters(), lr=cfg.lr_i)
    lr_func = lambda x: (1 - x / cfg.num_epochs) * (1 - cfg.lr_f) + cfg.lr_f
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    if cfg.resume is not None:
        logging.info(f'Resume from {cfg.resume}')
        checkpoint = torch.load(cfg.resume)
        fuse_net.load_state_dict(checkpoint['fuse_net'])

    loss_ssim = kornia.losses.SSIMLoss(window_size=11)
    loss_grad_pixel = PixelGradLoss()

    train_d = getattr(dataset, cfg.dataset_name)
    train_dataset = train_d(cfg, 'train')

    trainloader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=train_dataset.__collate_fn__, pin_memory=True
    )

    '''
    ------------------------------------------------------------------------------
    Train
    ------------------------------------------------------------------------------
    '''

    # torch.backends.cudnn.benchmark = True
    logging.info('Start training...')
    for epoch in range(cfg.start_epoch, cfg.num_epochs):
        '''train'''
        total_loss_meter = AverageMeter()
        content_loss_meter = AverageMeter()
        ssim_loss_meter = AverageMeter()
        saliency_loss_meter = AverageMeter()
        fre_loss_meter = AverageMeter()

        log_dict = {}
        loss_dict = {}
        iter = tqdm(trainloader, total=len(trainloader), ncols=80)
        for data_ir, data_vi, mask, _ in iter:
            data_ir, data_vi, mask = data_ir.to(device), data_vi.to(device), mask.to(device)
            for m in module_list:
                m.train()
            fus_data, amp, pha = fuse_net(data_ir, data_vi)
            # conten_loss
            content_loss = loss_grad_pixel(data_vi, data_ir, fus_data)

            # SSIM-loss
            ssim_loss_v = loss_ssim(data_vi, fus_data)
            ssim_loss_i = loss_ssim(data_ir, fus_data)
            ssim_loss = ssim_loss_i + ssim_loss_v

            # saliency_loss
            saliency_loss = cal_saliency_loss(fus_data, data_ir, data_vi, mask)

            # fre_loss
            fre_loss = cal_fre_loss(amp, pha, data_ir, data_vi, mask)

            total_loss = cfg.coeff_content * content_loss + cfg.coeff_ssim * ssim_loss + cfg.coeff_saliency * saliency_loss + cfg.coeff_fre * fre_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # loss dict
            loss_dict |= {
                'total_loss': total_loss.item(),
            }
            total_loss_meter.update(total_loss.item())
            content_loss_meter.update(content_loss.item())
            ssim_loss_meter.update(ssim_loss.item())
            saliency_loss_meter.update(saliency_loss.item())
            fre_loss_meter.update(fre_loss.item())
            # 设置进度条
            iter.set_description(f'Epoch {epoch + 1}/{cfg.num_epochs}')
            iter.set_postfix(loss_dict)

        scheduler.step()

        # 打印信息
        print('*' * 60 + '\tepoch finished!')
        logging.info(
            f'Epoch {epoch + 1}/{cfg.num_epochs}, lr:{optimizer.param_groups[0]["lr"]}, total_loss: {total_loss_meter.avg}, content_loss: {content_loss_meter.avg}, ssim_loss: {ssim_loss_meter.avg}, saliency_loss: {saliency_loss_meter.avg}, fre_loss: {fre_loss_meter.avg}'
        )

        log_dict |= {
            'total_loss': total_loss_meter.avg,
            'content_loss': content_loss_meter.avg,
            'ssim_loss': ssim_loss_meter.avg,
            'saliency_loss': saliency_loss_meter.avg,
            'fre_loss': fre_loss_meter.avg,
            'lr': optimizer.param_groups[0]["lr"],
        }

        # update wandb
        runs.log(log_dict)

        # 每隔几个epoch保存一次模型
        if (epoch + 1) % cfg.epoch_gap == 0:
            checkpoint = {'fuse_net': fuse_net.state_dict()}

            logging.info(f'Save checkpoint to models/{cfg.exp_name}.pth')
            save_path = os.path.join("models", f'{cfg.exp_name}.pth')
            if not os.path.exists('models'):
                os.makedirs('models')
            torch.save(checkpoint, save_path)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='configs/cfg.yaml', help='config file path')
    parser.add_argument('--auth', default='', help='wandb auth api key')
    args = parser.parse_args()
    train(args.cfg, args.auth)
    # 运行命令行代码
    os.system(f'nohup python val.py')
