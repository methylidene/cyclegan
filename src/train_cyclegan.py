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

# ================= 3. 网络架构定义 =================
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

# ================= 学习率衰减策略 =================
class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch
    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

# ================= 4. 训练主循环 =================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前运行设备: {device}")

    # ---------------- 🌟 路径与超参数配置区 ----------------
    DATA_ROOT = '/root/autodl-tmp/cyclegan/data/test02' 
    EXP_NAME = 'exp_02_visloc_full_100ep'
    
    OUTPUT_DIR = os.path.join('/root/autodl-tmp/cyclegan/output', EXP_NAME) 
    IMAGE_DIR = os.path.join(OUTPUT_DIR, 'images')
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')

    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    BATCH_SIZE = 4           # 若使用 4090 且显存充足，可尝试上调至 8
    EPOCHS = 100             # 总 Epoch 数
    DECAY_EPOCH = 50         # 前50个Epoch学习率不变，后50个线性衰减
    LR = 0.0002
    
    PRINT_INTERVAL = 400     # 每 400 个 Batch 打印一次控制台日志
    SAVE_IMG_EPOCH_FREQ = 1  # 每 1 个 Epoch 存一次预览图，防爆硬盘
    SAVE_EPOCH_FREQ = 20     # 每 20 个 Epoch 存一个永久权重节点
    # ------------------------------------------------------

    transforms_ = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataloader = DataLoader(UnpairedDataset(DATA_ROOT, transform=transforms_), 
                            batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=8, pin_memory=True)

    # 初始化网络
    G_AB = Generator().to(device) 
    G_BA = Generator().to(device) 
    D_A = Discriminator().to(device)
    D_B = Discriminator().to(device)

    # 优化器
    optimizer_G = optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=LR, betas=(0.5, 0.999))
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=LR, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=LR, betas=(0.5, 0.999))

    # ---------------- 🌟 断点续训读取 ----------------
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
        print(f"✅ 成功从 Epoch {start_epoch} 恢复。")

    # 初始化学习率调度器
    lr_scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(EPOCHS, start_epoch, DECAY_EPOCH).step)
    lr_scheduler_D_A = optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(EPOCHS, start_epoch, DECAY_EPOCH).step)
    lr_scheduler_D_B = optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(EPOCHS, start_epoch, DECAY_EPOCH).step)

    if os.path.exists(latest_ckpt_path) and 'lr_scheduler_G' in checkpoint:
        lr_scheduler_G.load_state_dict(checkpoint['lr_scheduler_G'])
        lr_scheduler_D_A.load_state_dict