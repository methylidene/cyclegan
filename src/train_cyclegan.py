import os
import random
import itertools
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image

# ================= 1. 数据集定义 =================
class UnpairedDataset(Dataset):
    def __init__(self, root_dir, phase='train', transform=None):
        self.dir_A = os.path.join(root_dir, f'{phase}A')
        self.dir_B = os.path.join(root_dir, f'{phase}B')
        
        self.A_paths = sorted([os.path.join(self.dir_A, f) for f in os.listdir(self.dir_A) if f.endswith(('png', 'jpg', 'jpeg', 'JPG', 'PNG'))])
        self.B_paths = sorted([os.path.join(self.dir_B, f) for f in os.listdir(self.dir_B) if f.endswith(('png', 'jpg', 'jpeg', 'JPG', 'PNG'))])
        
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform = transform

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[random.randint(0, self.B_size - 1)]
        
        img_A = Image.open(A_path).convert('RGB')
        img_B = Image.open(B_path).convert('RGB')
        
        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)
            
        return {'A': img_A, 'B': img_B}

    def __len__(self):
        return max(self.A_size, self.B_size)

# ================= 2. 图像缓冲池 =================
class ImageBuffer:
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.images = []

    def query(self, images):
        if self.pool_size == 0: 
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if len(self.images) < self.pool_size:
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return torch.cat(return_images, 0)

# ================= 3. 网络架构定义 (ResNet Generator & PatchGAN Discriminator) =================
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )
    def forward(self, x): return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, num_residual_blocks=9):
        super(Generator, self).__init__()
        model = [nn.ReflectionPad2d(3), nn.Conv2d(3, 64, 7), nn.InstanceNorm2d(64), nn.ReLU(inplace=True)]
        in_features, out_features = 64, 128
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features), nn.ReLU(inplace=True)]
            in_features, out_features = out_features, out_features * 2
        for _ in range(num_residual_blocks): model += [ResidualBlock(in_features)]
        out_features = in_features // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features), nn.ReLU(inplace=True)]
            in_features, out_features = out_features, out_features // 2
        model += [nn.ReflectionPad2d(3), nn.Conv2d(64, 3, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)
    def forward(self, x): return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, padding=1), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, padding=1)
        )
    def forward(self, x): return self.model(x)

# ================= 4. 训练主循环 =================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前运行设备: {device}")

    # ---------------- 🌟 路径与超参数配置区 ----------------
    DATA_ROOT = 'E:/A-academics/4_1/毕业设计/data/test01' 
    EXP_NAME = 'exp_01_baseline_local_test'
    
    OUTPUT_DIR = os.path.join('output', EXP_NAME) 
    IMAGE_DIR = os.path.join(OUTPUT_DIR, 'images')
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')

    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    BATCH_SIZE = 1 
    EPOCHS = 200     # 实际上限设置高一些
    LR = 0.0002
    SAMPLE_INTERVAL = 50 
    SAVE_EPOCH_FREQ = 50 # 每 50 个 Epoch 保存一个永久备份
    # ------------------------------------------------------

    transforms_ = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataloader = DataLoader(UnpairedDataset(DATA_ROOT, transform=transforms_), 
                            batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # 初始化网络
    G_AB = Generator().to(device) 
    G_BA = Generator().to(device) 
    D_A = Discriminator().to(device)
    D_B = Discriminator().to(device)

    # 优化器
    optimizer_G = optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=LR, betas=(0.5, 0.999))
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=LR, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=LR, betas=(0.5, 0.999))

    # ---------------- 🌟 存储优化型：断点续训读取 ----------------
    latest_ckpt_path = os.path.join(CHECKPOINT_DIR, "checkpoint_latest.pth")
    start_epoch = 0
    
    if os.path.exists(latest_ckpt_path):
        print(f"🔄 发现存档: {latest_ckpt_path}，正在恢复全量训练状态...")
        checkpoint = torch.load(latest_ckpt_path)
        
        G_AB.load_state_dict(checkpoint['G_AB'])
        G_BA.load_state_dict(checkpoint['G_BA'])
        D_A.load_state_dict(checkpoint['D_A'])
        D_B.load_state_dict(checkpoint['D_B'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        optimizer_D_A.load_state_dict(checkpoint['optimizer_D_A'])
        optimizer_D_B.load_state_dict(checkpoint['optimizer_D_B'])
        
        start_epoch = checkpoint['epoch'] + 1
        print(f"✅ 成功从 Epoch {start_epoch} 恢复，动量已对齐。")
    else:
        print("🆕 未发现存档，将从头开始训练。")
    # ------------------------------------------------------------

    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    fake_A_buffer = ImageBuffer()
    fake_B_buffer = ImageBuffer()

    print(f"🚀 实验 [{EXP_NAME}] 启动...")
    for epoch in range(start_epoch, EPOCHS):
        for i, batch in enumerate(dataloader):
            real_A, real_B = batch['A'].to(device), batch['B'].to(device)
            
            # 适配 256x256 输入的 PatchGAN 输出尺寸 (30x30)
            valid = torch.ones((real_A.size(0), 1, 30, 30), device=device, requires_grad=False)
            fake = torch.zeros((real_A.size(0), 1, 30, 30), device=device, requires_grad=False)

            # --- 训练生成器 ---
            optimizer_G.zero_grad()
            loss_id_A = criterion_identity(G_BA(real_A), real_A) * 5.0
            loss_id_B = criterion_identity(G_AB(real_B), real_B) * 5.0

            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

            recov_A, recov_B = G_BA(fake_B), G_AB(fake_A)
            loss_cycle_A = criterion_cycle(recov_A, real_A) * 10.0
            loss_cycle_B = criterion_cycle(recov_B, real_B) * 10.0

            loss_G = loss_GAN_AB + loss_GAN_BA + loss_cycle_A + loss_cycle_B + loss_id_A + loss_id_B
            loss_G.backward()
            optimizer_G.step()

            # --- 训练判别器 ---
            optimizer_D_A.zero_grad()
            loss_D_A = (criterion_GAN(D_A(real_A), valid) + criterion_GAN(D_A(fake_A_buffer.query(fake_A.detach())), fake)) * 0.5
            loss_D_A.backward()
            optimizer_D_A.step()

            optimizer_D_B.zero_grad()
            loss_D_B = (criterion_GAN(D_B(real_B), valid) + criterion_GAN(D_B(fake_B_buffer.query(fake_B.detach())), fake)) * 0.5
            loss_D_B.backward()
            optimizer_D_B.step()

            if i % SAMPLE_INTERVAL == 0:
                print(f"[Epoch {epoch}/{EPOCHS}] [Batch {i}/{len(dataloader)}] [D loss: {loss_D_A.item()+loss_D_B.item():.4f}] [G loss: {loss_G.item():.4f}]")
                image_sample = torch.cat((real_A.data, fake_B.data, real_B.data, fake_A.data), -2)
                save_image(image_sample, os.path.join(IMAGE_DIR, f"epoch_{epoch}_batch_{i}.png"), nrow=1, normalize=True)

        # ---------------- 🌟 存储优化型：打包与覆盖存档 ----------------
        checkpoint_state = {
            'epoch': epoch,
            'G_AB': G_AB.state_dict(), 'G_BA': G_BA.state_dict(),
            'D_A': D_A.state_dict(), 'D_B': D_B.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'optimizer_D_A': optimizer_D_A.state_dict(),
            'optimizer_D_B': optimizer_D_B.state_dict()
        }
        
        # 始终覆盖保存最新的状态
        torch.save(checkpoint_state, latest_ckpt_path)
        
        # 周期性备份，防止 latest 文件意外损坏
        if (epoch + 1) % SAVE_EPOCH_FREQ == 0:
            torch.save(checkpoint_state, os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth"))
            print(f"🚩 关键节点备份已完成: Epoch {epoch+1}")
            
        print(f"💾 Epoch {epoch} 训练进度已同步至最新存档。")

if __name__ == '__main__':
    main()