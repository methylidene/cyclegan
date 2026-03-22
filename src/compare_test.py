import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils

# ================= 1. 核心网络结构 (保持独立) =================
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


# ================= 2. 核心双向对比逻辑 =================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前运行设备: {device}")
    
    # ---------------- 🌟 极客配置区 ----------------
    # 1. 挑选测试图像 (A: 无人机, B: 卫星)
    TEST_IMAGE_PATH_A = '/root/autodl-tmp/cyclegan/data/test02/trainA/01_drone_01_0041.JPG' 
    TEST_IMAGE_PATH_B = '/root/autodl-tmp/cyclegan/data/test02/trainB/01_satellite01_patch_0127.jpg' 
    
    # 2. 权重路径
    BASELINE_CKPT = '/root/autodl-tmp/cyclegan/output/exp_02_visloc_full_100ep/checkpoints/checkpoint_epoch_60.pth' 
    V2_CKPT = '/root/autodl-tmp/cyclegan/output/exp_04_visloc_ssim_gray_100ep/checkpoints/checkpoint_epoch_60.pth' 
    
    # 3. 输出路径
    OUTPUT_DIR = '/root/autodl-tmp/cyclegan/output/exp_04_visloc_ssim_gray_100ep/'
    OUTPUT_COMPARE_A2B = os.path.join(OUTPUT_DIR, 'compare_A2B_epoch60_01.png')
    OUTPUT_COMPARE_B2A = os.path.join(OUTPUT_DIR, 'compare_B2A_epoch60_01.png')
    # -----------------------------------------------

    # 初始化 4 个生成器 
    # AB 代表 无人机 -> 卫星，BA 代表 卫星 -> 无人机
    G_AB_baseline = Generator().to(device)
    G_BA_baseline = Generator().to(device)
    G_AB_v2 = Generator().to(device)
    G_BA_v2 = Generator().to(device)
    
    # 切换到推理模式
    G_AB_baseline.eval(); G_BA_baseline.eval()
    G_AB_v2.eval(); G_BA_v2.eval()

    print("📥 正在加载 Baseline 双向权重...")
    checkpoint_base = torch.load(BASELINE_CKPT, map_location=device)
    G_AB_baseline.load_state_dict(checkpoint_base['G_AB'])
    G_BA_baseline.load_state_dict(checkpoint_base['G_BA'])

    print("📥 正在加载 V2 (SSIM) 双向权重...")
    checkpoint_v2 = torch.load(V2_CKPT, map_location=device)
    G_AB_v2.load_state_dict(checkpoint_v2['G_AB'])
    G_BA_v2.load_state_dict(checkpoint_v2['G_BA'])

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    print("🔍 正在读取测试图像...")
    img_A = Image.open(TEST_IMAGE_PATH_A).convert('RGB')
    img_B = Image.open(TEST_IMAGE_PATH_B).convert('RGB')
    
    tensor_A = transform(img_A).unsqueeze(0).to(device) 
    tensor_B = transform(img_B).unsqueeze(0).to(device) 

    # 🌟 关闭梯度跟踪，执行双向推理
    print("🚀 正在执行双向图像反演...")
    with torch.no_grad():
        # A -> B (无人机转卫星)
        fake_B_baseline = G_AB_baseline(tensor_A)
        fake_B_v2 = G_AB_v2(tensor_A)
        
        # B -> A (卫星转无人机)
        fake_A_baseline = G_BA_baseline(tensor_B)
        fake_A_v2 = G_BA_v2(tensor_B)

    # 将图片拼接在一起 (dim=3 是宽度方向拼接)
    compare_A2B_tensor = torch.cat((tensor_A.data, fake_B_baseline.data, fake_B_v2.data), dim=3)
    compare_B2A_tensor = torch.cat((tensor_B.data, fake_A_baseline.data, fake_A_v2.data), dim=3)
    
    # 保存结果
    vutils.save_image(compare_A2B_tensor, OUTPUT_COMPARE_A2B, nrow=1, normalize=True)
    vutils.save_image(compare_B2A_tensor, OUTPUT_COMPARE_B2A, nrow=1, normalize=True)
    
    print("\n===========================================")
    print(f"✅ [A -> B] 无人机转卫星对比图: {OUTPUT_COMPARE_A2B}")
    print(f"✅ [B -> A] 卫星转无人机对比图: {OUTPUT_COMPARE_B2A}")
    print("👉 图像从左到右顺序均是：[原图] | [Baseline(无SSIM)] | [V2(加了SSIM)]")
    print("===========================================")

if __name__ == '__main__':
    main()