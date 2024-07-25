from torch.utils.data import Dataset
import torch
from configs import *
import logging
from torchvision.transforms import Compose, Resize
from pathlib import Path
from typing import Literal
from utils.img_read import img_read
import os
from utils.saliency import Saliency


def check_mask(root: Path, img_list, config: ConfigDict):
    mask_cache = True
    if (root / 'mask').exists():
        for img_name in img_list:
            if not (root / 'mask' / img_name).exists():
                mask_cache = False
                break
    else:
        mask_cache = False
    if mask_cache:
        logging.info('find mask cache in folder, skip saliency detection')
    else:
        logging.info('find no mask cache in folder, start saliency detection')
        saliency = Saliency()
        saliency.inference(src=root / 'ir', dst=root / 'mask', suffix='png')


class M3FD(Dataset):
    def __init__(self, cfg: ConfigDict, mode: Literal['train', 'val', 'test']):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.img_list = Path(Path(self.cfg.dataset_root) / 'meta' / f'{mode}.txt').read_text().splitlines()
        logging.info(f'load {len(self.img_list)} images')
        self.train_transforms = Compose([Resize((cfg.img_size, cfg.img_size))])

        if self.mode == 'train' and cfg.have_seg_label == False:
            self.ir_path = Path(Path(self.cfg.dataset_root) / 'ir')
            self.vi_path = Path(Path(self.cfg.dataset_root) / 'vi')
            check_mask(Path(cfg.dataset_root), self.img_list, cfg)
            self.mask_path = Path(Path(self.cfg.dataset_root) / 'mask')
        if self.mode == 'train' and cfg.have_seg_label == True:
            self.ir_path = Path(Path(self.cfg.dataset_root) / 'ir')
            self.vi_path = Path(Path(self.cfg.dataset_root) / 'vi')
            self.mask_path = Path(Path(self.cfg.dataset_root) / 'labels')
        if self.mode == 'test':
            self.ir_path = Path(Path(self.cfg.dataset_root) / 'test' / 'ir')
            self.vi_path = Path(Path(self.cfg.dataset_root) / 'test' / 'vi')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_name = self.img_list[index]
        ir_img = img_read(os.path.join(self.ir_path, img_name), mode='L')
        vi_img, vi_cbcr = img_read(os.path.join(self.vi_path, img_name), mode='YCbCr')
        if self.mode == 'train':
            mask = img_read(os.path.join(self.mask_path, img_name), mode='L')
        else:
            mask = None

        if self.mode == 'train':
            ir_img = self.train_transforms(ir_img)
            vi_img = self.train_transforms(vi_img)
            mask = self.train_transforms(mask)
        else:
            _, h, w = ir_img.shape
            if h // 2 != 0 or w // 2 != 0:
                ir_img = ir_img[:, : h // 2 * 2, : w // 2 * 2]
                vi_img = vi_img[:, : h // 2 * 2, : w // 2 * 2]

        return ir_img, vi_img, mask, img_name

    def __collate_fn__(self, batch):
        ir_img_batch, vi_img_batch, mask_batch, img_name_batch = zip(*batch)
        ir_img_batch = torch.stack(ir_img_batch, dim=0)
        vi_img_batch = torch.stack(vi_img_batch, dim=0)
        if self.mode == 'train':
            mask_batch = torch.stack(mask_batch, dim=0)
        return ir_img_batch, vi_img_batch, mask_batch, img_name_batch


class MSRS(Dataset):
    def __init__(self, cfg: ConfigDict, mode: Literal['train', 'test']):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.img_list = os.listdir(os.path.join(cfg.dataset_root, self.mode, 'ir'))
        logging.info(f'load {len(self.img_list)} images')
        self.train_transforms = Compose([Resize((cfg.img_size, cfg.img_size))])

        self.ir_path = Path(Path(self.cfg.dataset_root) / self.mode / 'ir')
        self.vi_path = Path(Path(self.cfg.dataset_root) / self.mode / 'vi')
        if self.mode == 'train' and cfg.have_seg_label == False:
            check_mask(Path(cfg.dataset_root) / 'train', self.img_list, cfg)
            self.mask_path = Path(Path(self.cfg.dataset_root) / self.mode / 'mask')
        if self.mode == 'train' and cfg.have_seg_label == True:
            self.mask_path = Path(Path(self.cfg.dataset_root) / self.mode / 'labels')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_name = self.img_list[index]
        ir_img = img_read(os.path.join(self.ir_path, img_name), mode='L')
        vi_img, vi_cbcr = img_read(os.path.join(self.vi_path, img_name), mode='YCbCr')
        if self.mode == 'train':
            mask = img_read(os.path.join(self.mask_path, img_name), mode='L')
        else:
            mask = None

        if self.mode == 'train':
            ir_img = self.train_transforms(ir_img)
            vi_img = self.train_transforms(vi_img)
            mask = self.train_transforms(mask)
        else:
            _, h, w = ir_img.shape
            if h // 2 != 0 or w // 2 != 0:
                ir_img = ir_img[:, : h // 2 * 2, : w // 2 * 2]
                vi_img = vi_img[:, : h // 2 * 2, : w // 2 * 2]

        return ir_img, vi_img, mask, img_name

    def __collate_fn__(self, batch):
        ir_img_batch, vi_img_batch, mask_batch, img_name_batch = zip(*batch)
        ir_img_batch = torch.stack(ir_img_batch, dim=0)
        vi_img_batch = torch.stack(vi_img_batch, dim=0)
        if self.mode == 'train':
            mask_batch = torch.stack(mask_batch, dim=0)
        return ir_img_batch, vi_img_batch, mask_batch, img_name_batch


class RoadScene(Dataset):
    def __init__(self, cfg: ConfigDict, mode: Literal['train', 'val', 'test']):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.img_list = Path(Path(self.cfg.dataset_root) / f'{mode}.txt').read_text().splitlines()
        logging.info(f'load {len(self.img_list)} images')
        self.train_transforms = Compose([Resize((cfg.img_size, cfg.img_size))])

        if self.mode == 'train' and cfg.have_seg_label == False:
            self.ir_path = Path(Path(self.cfg.dataset_root) / 'ir')
            self.vi_path = Path(Path(self.cfg.dataset_root) / 'vi')
            check_mask(Path(cfg.dataset_root), self.img_list, cfg)
            self.mask_path = Path(Path(self.cfg.dataset_root) / 'mask')
        if self.mode == 'train' and cfg.have_seg_label == True:
            self.ir_path = Path(Path(self.cfg.dataset_root) / 'ir')
            self.vi_path = Path(Path(self.cfg.dataset_root) / 'vi')
            self.mask_path = Path(Path(self.cfg.dataset_root) / 'labels')
        if self.mode == 'test':
            self.ir_path = Path(Path(self.cfg.dataset_root) / 'test' / 'ir')
            self.vi_path = Path(Path(self.cfg.dataset_root) / 'test' / 'vi')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_name = self.img_list[index]
        ir_img = img_read(os.path.join(self.ir_path, img_name), mode='L')
        vi_img, vi_cbcr = img_read(os.path.join(self.vi_path, img_name), mode='YCbCr')
        if self.mode == 'train':
            mask = img_read(os.path.join(self.mask_path, img_name), mode='L')
        else:
            mask = None

        if self.mode == 'train':
            ir_img = self.train_transforms(ir_img)
            vi_img = self.train_transforms(vi_img)
            mask = self.train_transforms(mask)
        else:
            _, h, w = ir_img.shape
            if h // 2 != 0 or w // 2 != 0:
                ir_img = ir_img[:, : h // 2 * 2, : w // 2 * 2]
                vi_img = vi_img[:, : h // 2 * 2, : w // 2 * 2]

        return ir_img, vi_img, mask, img_name

    def __collate_fn__(self, batch):
        ir_img_batch, vi_img_batch, mask_batch, img_name_batch = zip(*batch)
        ir_img_batch = torch.stack(ir_img_batch, dim=0)
        vi_img_batch = torch.stack(vi_img_batch, dim=0)
        if self.mode == 'train':
            mask_batch = torch.stack(mask_batch, dim=0)
        return ir_img_batch, vi_img_batch, mask_batch, img_name_batch




if __name__ == '__main__':
    import yaml

    config = yaml.safe_load(open('./configs/cfg.yaml'))
    cfg = from_dict(config)
    train_dataset = MSRS(cfg, 'train')
    # 绘制数据集
    import matplotlib.pyplot as plt

    for i in range(3):
        ir, vi, mask, img_name = train_dataset[i]
        ir = ir.squeeze().numpy()
        vi = vi.squeeze().numpy()
        mask = mask.squeeze().numpy()
        plt.subplot(131)
        plt.imshow(ir, cmap='gray')
        plt.subplot(132)
        plt.imshow(vi, cmap='gray')
        plt.subplot(133)
        plt.imshow(mask, cmap='gray')
        plt.savefig(f'./{img_name}.png')
