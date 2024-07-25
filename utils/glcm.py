from skimage.feature import graycomatrix, graycoprops
import numpy as np
from kornia import image_to_tensor, tensor_to_image
import torch


def get_glcm(img, angles: list, dist=1, levels=256, symmetric=True, normed=True):
    '''
    img:[c, h, w]
    '''
    h, w = img.shape
    glcm = torch.zeros(len(angles), levels, levels)
    for i, angle in enumerate(angles):
        dx, dy = np.cos(angle), np.sin(angle)
        for y in range(h):
            for x in range(w):
                col, row = int(np.round(x + dx)), int(np.round(y + dy))
                if 0 <= col < w and 0 <= row < h:
                    glcm_clone = glcm.clone()
                    glcm_clone[i, img[y, x], img[row, col]] += 1
                    glcm = glcm_clone
    if symmetric:
        t = torch.transpose(glcm, 1, 2)
        gc = glcm.clone()
        gc += t
        glcm = gc
    if normed:
        glcm = torch.tensor(glcm, dtype=torch.float32)
        sums = torch.sum(glcm, dim=(1, 2), keepdim=True)
        sums[sums == 0] = 1
        glcm /= sums
    return glcm


# 实现计算glcm的特征
def calc_glcm_props(gm, prop):
    a, N, _ = gm.shape

    # 创建一个范围在[0, N-1]的张量
    I, J = torch.meshgrid(torch.arange(N), torch.arange(N))

    # 初始化结果张量
    result = torch.zeros(a)

    # 对每个角度分别计算属性
    for i in range(a):
        if prop == 'contrast':
            result[i] = torch.sum((I - J) ** 2 * gm[i])
        elif prop == 'ASM':
            result[i] = torch.sum(gm[i] ** 2)
        elif prop == 'homogeneity':
            result[i] = torch.sum(gm[i] / (1.0 + torch.abs(I - J)))
        elif prop == 'dissimilarity':
            result[i] = torch.sum(torch.abs(I - J) * gm[i])
        elif prop == 'correlation':
            I_mean = torch.sum(I * gm[i])
            J_mean = torch.sum(J * gm[i])
            I_std = torch.sqrt(torch.sum((I - I_mean) ** 2 * gm[i]))
            J_std = torch.sqrt(torch.sum((J - J_mean) ** 2 * gm[i]))
            result[i] = torch.sum((I - J_mean) * (J - J_mean) * gm[i]) / (I_std * J_std)
        else:
            raise ValueError('prop must be one of "contrast", "ASM", "homogeneity", "dissimilarity", "correlation"')

    return torch.mean(result)


def glcm_weight(ir, vi, dist=1, levels=256, symmetric=True, normed=True):
    ir = tensor_to_image(ir * 255.0, keepdim=True).astype(np.uint8)
    vi = tensor_to_image(vi * 255.0, keepdim=True).astype(np.uint8)
    n = ir.shape[0]
    ir_contrast = []
    ir_asm = []
    ir_homo = []
    ir_diss = []
    ir_corr = []
    vi_contrast = []
    vi_asm = []
    vi_homo = []
    vi_diss = []
    vi_corr = []
    for i in range(n):
        gm = graycomatrix(ir[i].squeeze(), [dist], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=levels, symmetric=symmetric, normed=normed)

        ir_contrast_value = graycoprops(gm, 'contrast')
        ir_asm_value = graycoprops(gm, 'ASM')
        ir_corr_value = graycoprops(gm, "correlation")
        ir_homo_value = graycoprops(gm, "homogeneity")
        ir_diss_value = graycoprops(gm, "dissimilarity")

        ir_contrast.append(ir_contrast_value)
        ir_asm.append(ir_asm_value)
        ir_corr.append(ir_corr_value)
        ir_homo.append(ir_homo_value)
        ir_diss.append(ir_diss_value)

        gm = graycomatrix(vi[i].squeeze(), [dist], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=levels, symmetric=symmetric, normed=normed)

        vi_contrast_value = graycoprops(gm, 'contrast')
        vi_asm_value = graycoprops(gm, 'ASM')
        vi_corr_value = graycoprops(gm, "correlation")
        vi_homo_value = graycoprops(gm, "homogeneity")
        vi_diss_value = graycoprops(gm, "dissimilarity")

        vi_contrast.append(vi_contrast_value)
        vi_asm.append(vi_asm_value)
        vi_corr.append(vi_corr_value)
        vi_homo.append(vi_homo_value)
        vi_diss.append(vi_diss_value)

    ir_contrast_value = np.mean(ir_contrast)
    ir_asm_value = np.mean(ir_asm)
    ir_corr_value = np.mean(ir_corr)
    ir_homo_value = np.mean(ir_homo)
    ir_diss_value = np.mean(ir_diss)

    vi_contrast_value = np.mean(vi_contrast)
    vi_asm_value = np.mean(vi_asm)
    vi_corr_value = np.mean(vi_corr)
    vi_homo_value = np.mean(vi_homo)
    vi_diss_value = np.mean(vi_diss)

    ir_homo_weight = ir_homo_value / (ir_homo_value + vi_homo_value)
    vi_homo_weight = 1 - ir_homo_weight

    ir_diss_weight = ir_diss_value / (ir_diss_value + vi_diss_value)
    vi_diss_weight = 1 - ir_diss_weight
    return ir_homo_weight + ir_diss_weight, vi_homo_weight + vi_diss_weight


if __name__ == '__main__':
    from img_read import img_read

    ir = img_read('../ir.png', 'L')
    vi, _ = img_read('../vi.png', 'YCbCr')
    a, b = glcm_weight(ir, vi)
    print(a, b)
