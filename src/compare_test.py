import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils

# ================= 1. 核心网络结构 (保持独立，防报错) =================
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(in_features, in_features, 3), nn.InstanceNorm2d(in_features), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(in_features, in_features, 3), nn.InstanceNorm2d(in_features)
        )
    def forward(self, x): return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, num_residual_blocks=9):
        super(Generator, self).__init__()
        model = [nn.ReflectionPad2d(3), nn.Conv2d(3, 64, 7), nn.InstanceNorm2d(64), nn.ReLU(inplace=True)]
        in_features, out_features = 64, 128
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1), nn.InstanceNorm2d(out_features), nn.ReLU(inplace=True)]
            in_features, out_features = out_features, out_features * 2
        for _ in range(num_residual_blocks): model += [ResidualBlock(in_features)]
        out_features = in_features // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1), nn.InstanceNorm2d(out_features), nn.ReLU(inplace=True)]
            in_features, out_features = out_features, out_features // 2
        model += [nn.ReflectionPad2d(3), nn.Conv2d(64, 3, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)
    def forward(self, x): return self.model(x)


# ================= 2. 核心对比逻辑 =================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ---------------- 🌟 极客配置区 ----------------
    # 1. 挑一张最能“找茬”的图：比如一半是森林，一半是建筑的无人机原图
    TEST_IMAGE_PATH = '/root/autodl-tmp/cyclegan/data/test02/trainA/填入你选好的图片名.jpg' 
    
    # 2. Baseline 的权重 (假设你想对比 Epoch 20 的状态)
    BASELINE_CKPT = '/root/autodl-tmp/cyclegan/output/exp_02_visloc_full_100ep/checkpoints/checkpoint_epoch_20.pth' 
    
    # 3. V2 (SSIM) 的权重 
    V2_CKPT = '/root/autodl-tmp/cyclegan/output/exp_03_visloc_ssim_100ep/checkpoints/checkpoint_epoch_20.pth' 
    
    # 4. 拼图输出路径
    OUTPUT_COMPARE_PATH = '/root/autodl-tmp/cyclegan/output/exp_03_visloc_ssim_100ep/compare_result_epoch20.png'
    # -----------------------------------------------

    # 初始化两个生成器 (A -> B，即 无人机 -> 卫星)
    G_baseline = Generator().to(device)
    G_v2 = Generator().to(device)
    
    # 🌟 极其重要：切换到推理模式 (冻结 Dropout 和 BatchNorm)
    G_baseline.eval()
    G_v2.eval()

    print("📥 正在加载 Baseline 权重...")
    checkpoint_base = torch.load(BASELINE_CKPT, map_location=device)
    G_baseline.load_state_dict(checkpoint_base['G_AB'])

    print("📥 正在加载 V2 (SSIM) 权重...")
    checkpoint_v2 = torch.load(V2_CKPT, map_location=device)
    G_v2.load_state_dict(checkpoint_v2['G_AB'])

    # 图像预处理 (和训练时保持绝对一致)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    print(f"🔍 正在读取测试图像: {TEST_IMAGE_PATH}")
    img = Image.open(TEST_IMAGE_PATH).convert('RGB')
    # unsqueeze(0) 是为了强行加上 Batch 维度，变成 [1, 3, 256, 256]
    input_tensor = transform(img).unsqueeze(0).to(device) 

    # 🌟 极其重要：关闭梯度跟踪，只做纯粹的矩阵乘法
    print("🚀 正在生成卫星图...")
    with torch.no_grad():
        fake_B_baseline = G_baseline(input_tensor)
        fake_B_v2 = G_v2(input_tensor)

    # 将三张图在宽度方向(dim=3)直接拼接到一起
    # 顺序：[真实无人机图] | [Baseline 假卫星图] | [V2 假卫星图]
    comparison_tensor = torch.cat((input_tensor.data, fake_B_baseline.data, fake_B_v2.data), dim=3)
    
    # 保存并自动解除归一化 (normalize=True 会自动把 [-1, 1] 拉回到 [0, 1])
    vutils.save_image(comparison_tensor, OUTPUT_COMPARE_PATH, nrow=1, normalize=True)
    print("\n===========================================")
    print(f"✅ 对比图已生成并保存至: {OUTPUT_COMPARE_PATH}")
    print("👉 图像从左到右顺序：[原图] | [无SSIM] | [加了SSIM]")
    print("===========================================")

if __name__ == '__main__':
    main()