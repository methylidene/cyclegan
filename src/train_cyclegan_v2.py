#引入ssim loss

import os
import random
import itertools
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils 
from torch.utils.tensorboard import SummaryWriter 

# 🌟 SSIM 专属新增：引入解包工具和第三方计算库
from utils import denormalize
from pytorch_msssim import SSIM

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

# ================= 3. 网络架构定义 =================
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

# ================= 学习率衰减策略 =================
class LambdaLR:
    def __init__(self, n_epochs, decay_start_epoch):
        self.n_epochs = n_epochs
        self.decay_start_epoch = decay_start_epoch
    def step(self, epoch):
        return max(0.0, 1.0 - max(0, epoch - self.decay_start_epoch) / float(self.n_epochs - self.decay_start_epoch))

# ================= 4. 训练主循环 =================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前运行设备: {device}")

    # ---------------- 🌟 路径与超参数配置区 ----------------
    # 强烈建议换一个新的输出目录，防止和 Baseline 的权重混淆！
    DATA_ROOT = '/root/autodl-tmp/cyclegan/data/test02' 
    EXP_NAME = 'exp_03_visloc_ssim_100ep'
    
    OUTPUT_DIR = os.path.join('/root/autodl-tmp/cyclegan/output', EXP_NAME) 
    IMAGE_DIR = os.path.join(OUTPUT_DIR, 'images')
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
    TB_LOG_DIR = os.path.join('/root/tf-logs', EXP_NAME)

    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(TB_LOG_DIR, exist_ok=True)

    BATCH_SIZE = 4           
    EPOCHS = 100             
    DECAY_EPOCH = 50         
    LR = 0.0002
    
    TB_LOG_INTERVAL = 100    
    PRINT_INTERVAL = 400     
    SAVE_IMG_EPOCH_FREQ = 1  
    SAVE_EPOCH_FREQ = 20     
    # ------------------------------------------------------

    writer = SummaryWriter(log_dir=TB_LOG_DIR)
    print(f"📊 TensorBoard 已启动，日志将保存在: {TB_LOG_DIR}")

    transforms_ = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataloader = DataLoader(UnpairedDataset(DATA_ROOT, transform=transforms_), 
                            batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=4, pin_memory=False)

    G_AB = Generator().to(device) 
    G_BA = Generator().to(device) 
    D_A = Discriminator().to(device)
    D_B = Discriminator().to(device)

    optimizer_G = optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=LR, betas=(0.5, 0.999))
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=LR, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=LR, betas=(0.5, 0.999))

    # ---------------- 断点续训读取 ----------------
    latest_ckpt_path = os.path.join(CHECKPOINT_DIR, "checkpoint_latest.pth")
    start_epoch = 0
    global_step = 0 
    
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
        global_step = start_epoch * len(dataloader) 
        print(f"✅ 成功从 Epoch {start_epoch} 恢复。")

    lr_scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(EPOCHS, DECAY_EPOCH).step)
    lr_scheduler_D_A = optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(EPOCHS, DECAY_EPOCH).step)
    lr_scheduler_D_B = optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(EPOCHS, DECAY_EPOCH).step)

    if os.path.exists(latest_ckpt_path) and 'lr_scheduler_G' in checkpoint:
        lr_scheduler_G.load_state_dict(checkpoint['lr_scheduler_G'])
        lr_scheduler_D_A.load_state_dict(checkpoint['lr_scheduler_D_A'])
        lr_scheduler_D_B.load_state_dict(checkpoint['lr_scheduler_D_B'])

    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    
    # 🌟 SSIM 专属新增：初始化结构相似度计算器
    # data_range=1.0 代表解包后的张量在 [0, 1] 之间
    criterion_SSIM = SSIM(data_range=1.0, size_average=True, channel=3).to(device)

    fake_A_buffer = ImageBuffer()
    fake_B_buffer = ImageBuffer()

    print(f"🚀 实验 [{EXP_NAME}] 启动... 总计 {EPOCHS} Epochs")
    
    try:
        for epoch in range(start_epoch, EPOCHS):
            for i, batch in enumerate(dataloader):
                real_A, real_B = batch['A'].to(device), batch['B'].to(device)
                
                valid = torch.ones((real_A.size(0), 1, 30, 30), device=device, requires_grad=False)
                fake = torch.zeros((real_A.size(0), 1, 30, 30), device=device, requires_grad=False)

                # --- 训练生成器 ---
                optimizer_G.zero_grad()
                
                loss_id_A = criterion_identity(G_BA(real_A), real_A) * 10.0
                loss_id_B = criterion_identity(G_AB(real_B), real_B) * 10.0

                fake_B = G_AB(real_A)
                loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
                fake_A = G_BA(real_B)
                loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

                recov_A, recov_B = G_BA(fake_B), G_AB(fake_A)
                loss_cycle_A = criterion_cycle(recov_A, real_A) * 10.0
                loss_cycle_B = criterion_cycle(recov_B, real_B) * 10.0

                # 🌟 SSIM 专属新增：计算结构封印 Loss
                # 注意 1：必须套上 denormalize 解包护盾
                # 注意 2：1.0 减去 SSIM 相似度，转化为最小化误差，权重设为 5.0
                loss_ssim_A = (1.0 - criterion_SSIM(denormalize(fake_B), denormalize(real_A))) * 5.0
                loss_ssim_B = (1.0 - criterion_SSIM(denormalize(fake_A), denormalize(real_B))) * 5.0

                # 将 SSIM 损失汇入总损失池
                loss_G = loss_GAN_AB + loss_GAN_BA + loss_cycle_A + loss_cycle_B + loss_id_A + loss_id_B + loss_ssim_A + loss_ssim_B
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

                # ---------------- 🌟 TensorBoard 核心监控打点 ----------------
                if global_step % TB_LOG_INTERVAL == 0:
                    current_lr = optimizer_G.param_groups[0]['lr']
                    
                    writer.add_scalar('Loss/Generator_Total', loss_G.item(), global_step)
                    writer.add_scalar('Loss/Discriminator_A', loss_D_A.item(), global_step)
                    writer.add_scalar('Loss/Discriminator_B', loss_D_B.item(), global_step)
                    
                    writer.add_scalars('Generator_Breakdown/GAN_Loss', {'AB': loss_GAN_AB.item(), 'BA': loss_GAN_BA.item()}, global_step)
                    writer.add_scalars('Generator_Breakdown/Cycle_Loss', {'A': loss_cycle_A.item(), 'B': loss_cycle_B.item()}, global_step)
                    writer.add_scalars('Generator_Breakdown/Identity_Loss', {'A': loss_id_A.item(), 'B': loss_id_B.item()}, global_step)
                    
                    # 🌟 SSIM 专属新增：将 SSIM 曲线打入 TensorBoard，方便实时监控
                    writer.add_scalars('Generator_Breakdown/SSIM_Loss', {'A': loss_ssim_A.item(), 'B': loss_ssim_B.item()}, global_step)
                    
                    writer.add_scalar('Learning_Rate', current_lr, global_step)

                # --- 控制台日志打印 ---
                if i % PRINT_INTERVAL == 0:
                    current_lr = optimizer_G.param_groups[0]['lr']
                    # 控制台上顺手把 SSIM 打印出来，心里有底
                    print(f"[Epoch {epoch}/{EPOCHS}] [Batch {i}/{len(dataloader)}] "
                          f"[D loss: {loss_D_A.item()+loss_D_B.item():.4f}] "
                          f"[G loss: {loss_G.item():.4f}] "
                          f"[SSIM loss: {loss_ssim_A.item()+loss_ssim_B.item():.4f}] "
                          f"[LR: {current_lr:.6f}]")

                global_step += 1

            # --- 保存预览图 ---
            if epoch % SAVE_IMG_EPOCH_FREQ == 0:
                image_sample = torch.cat((real_A.data, fake_B.data, real_B.data, fake_A.data), -2)
                
                save_path = os.path.join(IMAGE_DIR, f"epoch_{epoch:03d}.png")
                vutils.save_image(image_sample, save_path, nrow=1, normalize=True)
                
                img_grid = vutils.make_grid(image_sample, nrow=1, normalize=True)
                writer.add_image('VisLoc_Images/Real_A_|_Fake_B_|_Real_B_|_Fake_A', img_grid, global_step)
                
                print(f"🖼️ 已保存本轮预览图并同步至 TensorBoard: {save_path}")

            # --- 更新学习率 ---
            lr_scheduler_G.step()
            lr_scheduler_D_A.step()
            lr_scheduler_D_B.step()

            # --- 打包与覆盖存档 ---
            checkpoint_state = {
                'epoch': epoch,
                'G_AB': G_AB.state_dict(), 'G_BA': G_BA.state_dict(),
                'D_A': D_A.state_dict(), 'D_B': D_B.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D_A': optimizer_D_A.state_dict(),
                'optimizer_D_B': optimizer_D_B.state_dict(),
                'lr_scheduler_G': lr_scheduler_G.state_dict(),
                'lr_scheduler_D_A': lr_scheduler_D_A.state_dict(),
                'lr_scheduler_D_B': lr_scheduler_D_B.state_dict()
            }
            
            torch.save(checkpoint_state, latest_ckpt_path)
            if (epoch + 1) % SAVE_EPOCH_FREQ == 0:
                torch.save(checkpoint_state, os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth"))
                print(f"🚩 关键节点备份已完成: Epoch {epoch+1}")
                
            print(f"💾 Epoch {epoch} 训练进度已同步至最新存档。")

    except KeyboardInterrupt:
        print("\n[!] 检测到用户中断 (Ctrl+C)！正在将当前进度保存至安全存档...")
        checkpoint_state['epoch'] = epoch
        interrupt_ckpt_path = os.path.join(CHECKPOINT_DIR, "interrupt_backup.pth")
        torch.save(checkpoint_state, interrupt_ckpt_path)
        print(f"✅ 进度已安全保存至 {interrupt_ckpt_path}。你可以随时关机了。")
    finally:
        writer.close() 

if __name__ == '__main__':
    main()