from modules import *
import os
import numpy as np
from utils.evaluator import Evaluator
import torch
from utils.img_read import *
import argparse
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

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
log_f = '%(asctime)s | %(filename)s[line:%(lineno)d] | %(levelname)s | %(message)s'
logging.basicConfig(level='INFO', format=log_f)


def test(args):
    test_d = getattr(dataset, cfg.dataset_name)
    test_dataset = test_d(cfg, 'test')

    testloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=cfg.num_workers, collate_fn=test_dataset.__collate_fn__, pin_memory=True
    )
    fuse_out_folder = args.out_dir
    if not os.path.exists(fuse_out_folder):
        os.makedirs(fuse_out_folder)

    fuse_net = Fuse()
    ckpt = torch.load(args.ckpt_path, map_location=device)
    fuse_net.load_state_dict(ckpt['fuse_net'])
    fuse_net.to(device)
    fuse_net.eval()

    time_list = []
    with torch.no_grad():
        logging.info(f'fusing images ...')
        iter = tqdm(testloader, total=len(testloader), ncols=80)
        for data_ir, data_vi, _, img_name in iter:
            data_vi, data_ir = data_vi.to(device), data_ir.to(device)

            ts = time.time()
            fus_data, _, _ = fuse_net(data_ir, data_vi)
            te = time.time()
            time_list.append(te - ts)
            if args.mode == 'gray':
                fi = np.squeeze((fus_data * 255).cpu().numpy()).astype(np.uint8)
                img_save(fi, img_name[0], fuse_out_folder)
            elif args.mode == 'RGB':
                vi_cbcr = vi_cbcr.to(device)
                fi = torch.cat((fus_data, vi_cbcr), dim=1)
                fi = ycbcr_to_rgb(fi)
                fi = tensor_to_image(fi) * 255
                fi = fi.astype(np.uint8)
                img_save(fi, img_name[0], fuse_out_folder, mode='RGB')

    logging.info(f'fusing images done!')
    logging.info(f'time: {np.round(np.mean(time_list[1:]), 6)}s')
    evaluate(fuse_out_folder)


def evaluate(fuse_out_folder):
    test_d = getattr(dataset, cfg.dataset_name)
    test_dataset = test_d(cfg, 'test')
    testloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=cfg.num_workers, collate_fn=test_dataset.__collate_fn__, pin_memory=True
    )
    metric_result = [AverageMeter() for _ in range(6)]

    logging.info(f'evaluating images ...')
    iter = tqdm(testloader, total=len(testloader), ncols=80)

    for data_ir, data_vi, _, img_name in iter:
        ir = data_ir.numpy().squeeze() * 255
        vi = data_vi.numpy().squeeze() * 255
        fi = img_read(os.path.join(fuse_out_folder, img_name[0]), 'L').numpy().squeeze() * 255
        h, w = fi.shape
        if h // 2 != 0 or w // 2 != 0:
            fi = fi[: h // 2 * 2, : w // 2 * 2]
        if fi.shape != ir.shape or fi.shape != vi.shape:
            fi = cv2.resize(fi, (ir.shape[1], ir.shape[0]))
        # print(ir.shape, vi.shape, fi.shape)
        metric_result[0].update(Evaluator.EN(fi))
        metric_result[1].update(Evaluator.SD(fi))
        metric_result[2].update(Evaluator.SF(fi))
        metric_result[3].update(Evaluator.MI(fi, ir, vi))
        metric_result[4].update(Evaluator.VIFF(fi, ir, vi))
        metric_result[5].update(Evaluator.Qabf(fi, ir, vi))

    # 结果写入文件
    with open(f'{fuse_out_folder}_result.txt', 'w') as f:
        f.write('EN: ' + str(np.round(metric_result[0].avg, 3)) + '\n')
        f.write('SD: ' + str(np.round(metric_result[1].avg, 3)) + '\n')
        f.write('SF: ' + str(np.round(metric_result[2].avg, 3)) + '\n')
        f.write('MI: ' + str(np.round(metric_result[3].avg, 3)) + '\n')
        f.write('VIF: ' + str(np.round(metric_result[4].avg, 3)) + '\n')
        f.write('Qabf: ' + str(np.round(metric_result[5].avg, 3)) + '\n')

    logging.info(f'writing results done!')
    print("\n" * 2 + "=" * 80)
    print("The test result :")
    print("\t\t EN\t SD\t SF\t MI\tVIF\tQabf")
    print(
        'result:\t'
        + '\t'
        + str(np.round(metric_result[0].avg, 3))
        + '\t'
        + str(np.round(metric_result[1].avg, 3))
        + '\t'
        + str(np.round(metric_result[2].avg, 3))
        + '\t'
        + str(np.round(metric_result[3].avg, 3))
        + '\t'
        + str(np.round(metric_result[4].avg, 3))
        + '\t'
        + str(np.round(metric_result[5].avg, 3))
    )
    print("=" * 80)


if __name__ == "__main__":
    config = yaml.safe_load(open('configs/cfg.yaml'))
    cfg = from_dict(config)
    parse = argparse.ArgumentParser()
    parse.add_argument('--ckpt_path', type=str, default=f'models/{cfg.exp_name}.pth')
    parse.add_argument('--dataset_name', type=str, default=cfg.dataset_name)
    parse.add_argument('--out_dir', type=str, default=f'test_result/{cfg.dataset_name}/{cfg.exp_name}')
    parse.add_argument('--mode', type=str, default='gray')
    args = parse.parse_args()

    test(args)
    # evaluate("./test_result/res.txt")
