# pip install pytorch-fid lpips

import os
import torch
import lpips
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm


from train_cyclegan_v2_1 import Generator 

def denormalize(tensor):
    return (tensor + 1.0) / 2.0

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 启动终极评估引擎，当前设备: {device}")

    # ================= 1. 核心配置区 =================
    # 测试集路径 (A: 无人机原图, B: 真实的卫星图)
    TEST_A_DIR = '/root/autodl-tmp/cyclegan/data/test02/testA' 
    TEST_B_DIR = '/root/autodl-tmp/cyclegan/data/test02/testB'
    
    # 你的 V2 版本权重路径 (等跑到 Epoch 100 的时候填这里)
    WEIGHTS_PATH = '/root/autodl-tmp/cyclegan/output/exp_04_visloc_ssim_gray_100ep/checkpoints/checkpoint_epoch_100.pth'
    
    # 生成的假卫星图存放目录 (为了算 FID，必须把图片都存下来)
    FAKE_B_DIR = '/root/autodl-tmp/cyclegan/output/evaluation/fake_B/exp_04_visloc_ssim_gray_100ep'
    os.makedirs(FAKE_B_DIR, exist_ok=True)
    # =================================================

    # 1. 加载生成器
    G_AB = Generator().to(device)
    G_AB.eval() # 极其重要：关闭 BatchNorm 的动态计算
    
    print("📥 正在加载生成器权重...")
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    G_AB.load_state_dict(checkpoint['G_AB'])

    # 2. 初始化 LPIPS 感知距离计算器 (底层使用 VGG 网络提取特征)
    print("🧠 正在加载 LPIPS-VGG 深度特征感知模型...")
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

    # 图像预处理 (和训练时必须绝对一致)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 获取所有测试图片
    img_names = [f for f in os.listdir(TEST_A_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    total_lpips_distance = 0.0
    
    print(f"🏃 开始批量推理并计算 LPIPS，共计 {len(img_names)} 张图片...")
    with torch.no_grad(): # 极其重要：节约显存
        for img_name in tqdm(img_names):
            img_path = os.path.join(TEST_A_DIR, img_name)
            img_A = Image.open(img_path).convert('RGB')
            
            # [1, 3, 256, 256]
            tensor_A = transform(img_A).unsqueeze(0).to(device) 
            
            # 生成假卫星图
            fake_B = G_AB(tensor_A)
            
            # 🌟 计算 LPIPS (比较无人机原图和假卫星图的“感知结构差异”)
            # LPIPS 能够穿透颜色的表象，直接对比它们在 VGG 里的深层特征是不是一回事
            lpips_val = loss_fn_vgg(tensor_A, fake_B)
            total_lpips_distance += lpips_val.item()
            
            # 保存假卫星图到硬盘，准备给 FID 算总账
            save_path = os.path.join(FAKE_B_DIR, img_name)
            vutils.save_image(fake_B, save_path, normalize=True)

    # 计算平均 LPIPS
    avg_lpips = total_lpips_distance / len(img_names)
    print("\n" + "="*50)
    print(f"🎯 [阶段一完成] 平均 LPIPS 感知距离: {avg_lpips:.4f}")
    print("💡 提示：LPIPS 越低，说明结构保留得越好，幻觉越少！")
    print("="*50 + "\n")

    # ================= 3. 调用外部命令计算 FID =================
    print("📈 正在计算全局 FID (Fréchet Inception Distance)...")
    print(f"👉 对比真实数据: {TEST_B_DIR}")
    print(f"👉 对比生成数据: {FAKE_B_DIR}")
    print("这可能需要几分钟时间提取 Inception 特征，请耐心等待...\n")
    
    # 使用 os.system 直接在终端里调用 pytorch-fid 命令
    fid_command = f"python -m pytorch_fid {TEST_B_DIR} {FAKE_B_DIR} --device cuda"
    os.system(fid_command)

if __name__ == '__main__':
    main()