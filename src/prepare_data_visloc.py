import os
import glob
import cv2
import numpy as np
from tqdm import tqdm

def process_visloc_dataset(raw_root, out_root, patch_size=256, sat_stride=150, max_sat_per_folder=800):
    """
    UAV-VISLOC 全量数据预处理：
    - 前 9 个区域 (01-09) 作为训练集 (trainA, trainB)
    - 后 2 个区域 (10-11) 作为测试集 (testA, testB) 做到地理隔离
    """
    # 创建需要的输出目录
    dirs = ['trainA', 'trainB', 'testA', 'testB']
    for d in dirs:
        os.makedirs(os.path.join(out_root, d), exist_ok=True)

    sub_dirs = []
    for d in glob.glob(os.path.join(raw_root, '*')):
        if os.path.isdir(d):
            folder_name = os.path.basename(d)
            # 严格判断：只有文件夹名是纯数字（如 '01', '02', '11'）才加入处理列表
            if folder_name.isdigit():
                sub_dirs.append(d)
                
    sub_dirs = sorted(sub_dirs)
    
    if not sub_dirs:
        print(f"❌ 未找到纯数字命名的区域文件夹，请检查 RAW_DATASET_ROOT")
        return

    print(f"🔍 发现 {len(sub_dirs)} 个区域文件夹，准备全量处理...\n")
    stats = {'trainA': 0, 'trainB': 0, 'testA': 0, 'testB': 0}

    for sub_dir in sub_dirs:
        folder_name = os.path.basename(sub_dir) # 如: '01'
        
        #直接把纯数字字符串转为整数
        region_id = int(folder_name)
        
        if region_id <= 9:
            phase = 'train'
        else:
            phase = 'test'
            
        dir_A = os.path.join(out_root, f"{phase}A")
        dir_B = os.path.join(out_root, f"{phase}B")
        print(f"▶️ 处理 {folder_name} -> 划分至 [{phase}集]")

        # ================= 1. 处理 Drone 图像 (全量提取) =================
        drone_dir = os.path.join(sub_dir, 'drone')
        if os.path.exists(drone_dir):
            drone_imgs = [os.path.join(drone_dir, f) for f in os.listdir(drone_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_path in tqdm(drone_imgs, desc=f"  [Drone]", leave=False):
                img = cv2.imread(img_path)
                if img is not None:
                    img_resized = cv2.resize(img, (patch_size, patch_size))
                    save_name = f"{folder_name}_drone_{os.path.basename(img_path)}"
                    cv2.imwrite(os.path.join(dir_A, save_name), img_resized)
                    stats[f'{phase}A'] += 1

        # # ================= 2. 处理 Satellite TIF 图像 =================
        # tif_files = glob.glob(os.path.join(sub_dir, '*.tif'))
        # for tif_path in tif_files:
        #     tif_name = os.path.splitext(os.path.basename(tif_path))[0]
        #     img_sat = cv2.imread(tif_path, cv2.IMREAD_UNCHANGED)
        #     if img_sat is None: continue
                
        #     if img_sat.dtype != np.uint8:
        #         img_sat = cv2.normalize(img_sat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        #     if len(img_sat.shape) == 2:
        #         img_sat = cv2.cvtColor(img_sat, cv2.COLOR_GRAY2BGR)

        #     h, w = img_sat.shape[:2]
        #     patch_count = 0
            
        #     # 使用略大的 stride 或限制数量，避免单张卫星图产出过多无用重叠
        #     with tqdm(total=max_sat_per_folder, desc=f"  [Satellite]", leave=False) as pbar:
        #         for y in range(0, h - patch_size + 1, sat_stride):
        #             for x in range(0, w - patch_size + 1, sat_stride):
        #                 if phase == 'train' and patch_count >= max_sat_per_folder:
        #                     break
                        
        #                 patch = img_sat[y:y+patch_size, x:x+patch_size]
                        
        #                 # 标准差过滤：剔除全黑边或大面积单一色彩（如死水、白云）
        #                 if np.std(patch) < 15.0: 
        #                     continue
                            
        #                 save_name = f"{folder_name}_{tif_name}_patch_{patch_count:04d}.jpg"
        #                 cv2.imwrite(os.path.join(dir_B, save_name), patch)
        #                 patch_count += 1
        #                 stats[f'{phase}B'] += 1
        #                 pbar.update(1)
                        
        #             if phase == 'train' and patch_count >= max_sat_per_folder:
        #                 break


        # ================= 2. 处理 Satellite TIF 图像 (超低内存版) =================
        import rasterio
        from rasterio.windows import Window
        
        tif_files = glob.glob(os.path.join(sub_dir, '*.tif'))
        for tif_path in tif_files:
            tif_name = os.path.splitext(os.path.basename(tif_path))[0]
            
            try:
                # 使用 rasterio 打开超大 TIF，此时【不】会加载图像像素到内存
                with rasterio.open(tif_path) as src:
                    h, w = src.height, src.width
                    patch_count = 0
                    
                    with tqdm(total=max_sat_per_folder, desc=f"  [Satellite]", leave=False) as pbar:
                        for y in range(0, h - patch_size + 1, sat_stride):
                            for x in range(0, w - patch_size + 1, sat_stride):
                                if phase == 'train' and patch_count >= max_sat_per_folder:
                                    break
                                
                                # 核心：只从硬盘读取当前坐标下的 256x256 小窗口
                                window = Window(x, y, patch_size, patch_size)
                                patch = src.read(window=window)
                                
                                # rasterio 读取的格式是 (Channel, Height, Width)，需要转成 OpenCV 习惯的 (H, W, C)
                                patch = np.transpose(patch, (1, 2, 0))
                                
                                # 提取 RGB 通道并转换为 BGR (OpenCV 的默认保存格式)
                                if patch.shape[2] >= 3:
                                    patch = patch[:, :, :3]
                                    patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
                                elif patch.shape[2] == 1:
                                    patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
                                
                                # 归一化高位深图像 (如 16-bit 降至 8-bit)
                                if patch.dtype != np.uint8:
                                    patch = cv2.normalize(patch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                                
                                # 剔除无效或大片单一颜色的色块
                                if np.std(patch) < 15.0: 
                                    continue
                                    
                                save_name = f"{folder_name}_{tif_name}_patch_{patch_count:04d}.jpg"
                                cv2.imwrite(os.path.join(dir_B, save_name), patch)
                                patch_count += 1
                                stats[f'{phase}B'] += 1
                                pbar.update(1)
                                
                            if phase == 'train' and patch_count >= max_sat_per_folder:
                                break
            except Exception as e:
                print(f"  ❌ 读取 TIF 失败 {tif_path}: {e}")

    print("\n✅ 数据预处理全部完成！")
    print(f"📊 统计信息: ")
    print(f"   训练集: 航拍(trainA) {stats['trainA']}张, 卫星(trainB) {stats['trainB']}张")
    print(f"   测试集: 航拍(testA) {stats['testA']}张, 卫星(testB) {stats['testB']}张")

if __name__ == "__main__":
    # 配置你的路径
    RAW_DATASET_ROOT = "/root/autodl-tmp/cyclegan/data/uav_visloc"
    OUTPUT_DATASET_ROOT = "/root/autodl-tmp/cyclegan/data/test02"
    
    process_visloc_dataset(
        raw_root=RAW_DATASET_ROOT,
        out_root=OUTPUT_DATASET_ROOT,
        patch_size=256,
        sat_stride=150,               
        max_sat_per_folder=800  # 9个训练文件夹 * 800 ≈ 7200张卫星图，与无人机图数量完美匹配
    )