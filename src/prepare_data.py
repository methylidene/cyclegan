import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def process_u1652_subset_by_class(source_dir, target_dir, num_classes=50):
    """
    按建筑类别(Class ID)抽取子集，保证 A域 和 B域 的物理地点完全对应
    """
    source_dir, target_dir = Path(source_dir), Path(target_dir)
    for d in ['trainA', 'trainB', 'testA', 'testB']:
        (target_dir / d).mkdir(parents=True, exist_ok=True)

    # 1. 获取所有可用的建筑类别 ID (例如 '0001', '0002')
    train_drone_dir = source_dir / 'train' / 'drone'
    if not train_drone_dir.exists():
        print("未找到源训练目录，请检查路径！")
        return
        
    all_classes = [d.name for d in train_drone_dir.iterdir() if d.is_dir()]
    
    # 2. 随机抽取指定数量的类别
    random.seed(42) # 固定种子，保证每次抽样的建筑一样
    if num_classes and num_classes < len(all_classes):
        selected_classes = set(random.sample(all_classes, num_classes))
        print(f"🎯 已随机抽取 {num_classes} 个建筑地点的全量视角进行组装...")
    else:
        selected_classes = set(all_classes)
        print("ℹ️ 使用全量建筑类别...")

    # 3. 开始按映射关系拷贝（只拷贝被选中的建筑）
    mapping = {
        source_dir / 'train' / 'drone': target_dir / 'trainA',
        source_dir / 'train' / 'satellite': target_dir / 'trainB',
        source_dir / 'test' / 'query_drone': target_dir / 'testA',
        source_dir / 'test' / 'gallery_satellite': target_dir / 'testB'
    }

# 涵盖遥感数据集最常见的所有后缀及大小写变体
    exts = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']

    for src_path, dst_path in mapping.items():
        if not src_path.exists(): continue
        
        # 步骤1: 用 rglob 像吸尘器一样把该目录下所有深度的图片全吸出来
        all_images = []
        for ext in exts:
            all_images.extend(list(src_path.rglob(f'*{ext}')))

        all_images = list(set(all_images))
            
        # 步骤2: 筛选符合抽样条件的图片
        images_to_copy = []
        for img_path in all_images:
            # 获取图片所属的建筑类别 ID (比如 0001)
            class_id = img_path.parent.name
            
            if 'train' in dst_path.name:
                if class_id in selected_classes:
                    images_to_copy.append(img_path)
            else:
                # 测试集的数据我们一律全量保留
                images_to_copy.append(img_path)

        images_to_copy = [img for img in images_to_copy if not img.name.startswith('._')]

        # 步骤3: 开始拷贝并显示进度条
        for img_path in tqdm(images_to_copy, desc=f"构建 {dst_path.name}"):
            class_id = img_path.parent.name
            new_name = f"{class_id}_{img_path.name}"
            shutil.copy(img_path, dst_path / new_name)

    print(f"\n✅ 基于 {num_classes} 个地点的子集构建完成！")

if __name__ == '__main__':
    SOURCE = 'E:/A-academics/4_1/毕业设计/data/University-Release' 
    TARGET = 'E:/A-academics/4_1/毕业设计/data/test01' 
    
    # 【核心参数】你想用多少栋建筑来做前期测试？
    # U-1652 训练集总共有 701 栋建筑。前期测试设为 50 或 100 即可。
    NUM_CLASSES_TO_SAMPLE = 50 
    
    process_u1652_subset_by_class(SOURCE, TARGET, num_classes=NUM_CLASSES_TO_SAMPLE)