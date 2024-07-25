import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PixelGradLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_vis, image_ir, fus_img):
        image_y = image_vis[:, :1, :, :]
        x_in_max = torch.max(image_y, image_ir)
        loss_in = F.l1_loss(x_in_max, fus_img)
        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        fus_img_grad = self.sobelconv(fus_img)
        x_grad_joint = torch.max(y_grad, ir_grad)
        loss_grad = F.l1_loss(x_grad_joint, fus_img_grad)
        return 5 * loss_in + 10 * loss_grad


class Sobelxy(nn.Module):
    def __init__(self):
        super().__init__()
        kernelx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        kernely = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)


def cal_saliency_loss(fus, ir, vi, mask):
    loss_tar = F.l1_loss(fus * mask, ir * mask)
    loss_back = F.l1_loss(fus * (1 - mask), vi * (1 - mask))
    return 5 * loss_tar + loss_back


# def cal_fre_loss(amp, pha, ir, vi):
#     real = amp * torch.cos(pha) + 1e-8
#     imag = amp * torch.sin(pha) + 1e-8
#     x = torch.complex(real, imag)
#     x = torch.abs(torch.fft.irfftn(x, dim=(-2, -1)))
#     x_max = torch.max(ir, vi)
#     res = F.l1_loss(x, x_max)
#     return res


def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1**2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)))
    cc = torch.clamp(cc, -1.0, 1.0)
    return cc.mean()


def cal_fre_loss(amp, pha, ir, vi, mask):
    real = amp * torch.cos(pha) + 1e-8
    imag = amp * torch.sin(pha) + 1e-8
    x = torch.complex(real, imag)
    x = torch.abs(torch.fft.irfftn(x, dim=(-2, -1)))
    loss_ir = cc(x * mask, ir * mask)
    loss_vi = cc(x * (1 - mask), vi * (1 - mask))
    return loss_ir + loss_vi
