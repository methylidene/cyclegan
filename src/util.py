# 记得pip install pytorch-msssim

import torch

def denormalize(tensor):
    """将 Generator 输出的 [-1, 1] 张量还原为 [0, 1]，专供 SSIM 计算使用"""
    return (tensor + 1.0) / 2.0
