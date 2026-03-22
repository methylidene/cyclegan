# 记得pip install pytorch-msssim

import torch

def denormalize(tensor):
    """将 Generator 输出的 [-1, 1] 张量还原为 [0, 1]，专供 SSIM 计算使用"""
    return (tensor + 1.0) / 2.0

def rgb_to_gray(tensor):
    """将 (B, 3, H, W) 的 RGB 张量转换为 (B, 1, H, W) 的灰度张量"""
    # 经典公式: Y = 0.299*R + 0.587*G + 0.114*B
    r, g, b = tensor[:, 0:1, :, :], tensor[:, 1:2, :, :], tensor[:, 2:3, :, :]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray