import cv2
import os
import shutil
import numpy as np
import random
from tqdm import tqdm

# ================= 核心算法 =================
def calculate_blur_variance(image_path):
    """计算图像的拉普拉斯方差，值越小代表图像越平坦/纯色 (抓水面)"""
    image = cv2.imread(image_path)
    if image is None:
        return -1
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def calculate_vov(image_path, grid_size=4):
    """计算方差的方差 (VoV)，值越小代表图像纹理越单一 (抓树冠)"""
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        return -1
        
    h, w = img_gray.shape
    patch_h, patch_w = h // grid_size, w // grid_size
    patch_variances = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            patch = img_gray[i*patch_h : (i+1)*patch_h, j*patch_w : (j+1)*patch_w]
            var = cv2.Laplacian(patch, cv2.CV_64F).var()
            patch_variances.append(var)
            
    # 取这 16 个区块方差的标准差，反映宏观异质性
    return np.std(patch_variances)

# ================= 第一步：试算分布 (双规探底) =================
def step1_analyze_distribution(source_dir, sample_size=1000):
    """随机抽取一批图片，同时计算方差和 VoV 分布"""
    images = [f for f in os.listdir(source_dir) if f.endswith(('.png', '.jpg','.JPG'))]
    sample_images = random.sample(images, min(sample_size, len(images)))
    
    print(f"\n🔍 [Step 1] 正在对 {len(sample_images)} 张图片进行 双指标分布 试算...")
    variances = []
    vovs = []
    
    for img_name in tqdm(sample_images):
        img_path = os.path.join(source_dir, img_name)
        var = calculate_blur_variance(img_path)
        vov = calculate_vov(img_path)
        if var != -1 and vov != -1:
            variances.append(var)
            vovs.append(vov)
            
    variances = np.array(variances)
    vovs = np.array(vovs)
    
    print("\n📊 === 数据集双指标分布报告 ===")
    print(f"{'指标':<15} | {'Min':<8} | {'10%分位':<8} | {'25%分位':<8} | {'中位数':<8} | {'Max':<8}")
    print("-" * 65)
    print(f"{'全局方差(Var)':<13} | {np.min(variances):<8.1f} | {np.percentile(variances, 10):<8.1f} | {np.percentile(variances, 25):<8.1f} | {np.median(variances):<8.1f} | {np.max(variances):<8.1f}")
    print(f"{'异质性(VoV)':<15} | {np.min(vovs):<8.1f} | {np.percentile(vovs, 10):<8.1f} | {np.percentile(vovs, 25):<8.1f} | {np.median(vovs):<8.1f} | {np.max(vovs):<8.1f}")
    print("=====================================\n")
    print("💡 提示：重点关注 VoV 的 10% 或 25% 分位数，低于这个值的很可能全是纹理单一的树冠！")
    return variances, vovs

# ================= 第二步：抽样盲测 (试效果) =================
def step2_sample_effect(source_dir, var_threshold, vov_threshold, sample_output_dir, num_samples=100):
    """根据你给定的双阈值，进行分类抽样"""
    good_sample_dir = os.path.join(sample_output_dir, 'sample_good')
    bad_sample_dir = os.path.join(sample_output_dir, 'sample_bad')
    
    if os.path.exists(sample_output_dir):
        shutil.rmtree(sample_output_dir)
    os.makedirs(good_sample_dir, exist_ok=True)
    os.makedirs(bad_sample_dir, exist_ok=True)

    images = [f for f in os.listdir(source_dir) if f.endswith(('.png', '.jpg','.JPG'))]
    sample_images = random.sample(images, min(num_samples, len(images)))
    
    print(f"\n🧪 [Step 2] 正在抽样测试 (Var < {var_threshold} 或 VoV < {vov_threshold} 视为废片)...")
    bad_count = 0
    for img_name in tqdm(sample_images):
        img_path = os.path.join(source_dir, img_name)
        var = calculate_blur_variance(img_path)
        vov = calculate_vov(img_path)
        
        # 🌟 名字上同时挂载两个指标：var_450_vov_120_DJI001.jpg
        new_name = f"var{var:.0f}_vov{vov:.0f}_{img_name}" 
        
        # 只要触碰任何一条红线，直接打入冷宫
        if var < var_threshold or vov < vov_threshold:
            shutil.copy(img_path, os.path.join(bad_sample_dir, new_name))
            bad_count += 1
        else:
            shutil.copy(img_path, os.path.join(good_sample_dir, new_name))
            
    print(f"✅ 抽样完成！拦截了 {bad_count}/{len(sample_images)} 张废片 (占比 {bad_count/len(sample_images)*100:.1f}%)。")
    print(f"👉 请去 {sample_output_dir} 查看结果，重点看那些 Var 很高但 VoV 很低的图是不是森林！\n")

# ================= 第三步：全量过滤 (动真格) =================
def step3_full_filter(source_dir, good_dir, bad_dir, final_var_thresh, final_vov_thresh):
    """确认阈值无误后，对全量数据集进行终极清洗"""
    os.makedirs(good_dir, exist_ok=True)
    os.makedirs(bad_dir, exist_ok=True)

    images = [f for f in os.listdir(source_dir) if f.endswith(('.png', '.jpg','.JPG'))]
    
    print(f"\n🚀 [Step 3] 最终清洗启动！(过滤条件: Var < {final_var_thresh} 或 VoV < {final_vov_thresh})")
    for img_name in tqdm(images):
        img_path = os.path.join(source_dir, img_name)
        var = calculate_blur_variance(img_path)
        vov = calculate_vov(img_path)
        
        if var < final_var_thresh or vov < final_vov_thresh:
            shutil.copy(img_path, os.path.join(bad_dir, img_name))
        else:
            shutil.copy(img_path, os.path.join(good_dir, img_name))
    print(f"🎉 遥感特征空间彻底净化完成！极其纯净的训练数据已存入: {good_dir}\n")


if __name__ == '__main__':
    # 你需要清洗的数据集路径
    INPUT_DIR = '/root/autodl-tmp/cyclegan/data/test02/trainA'
    
    # ---------------- 流程控制台 ----------------
    
    # 【执行阶段 1】：探底细。看看两个指标各自的分布
    step1_analyze_distribution(INPUT_DIR, sample_size=2000)

    # 【执行阶段 2】：试效果。
    # 假设你从 Step 1 发现：Var的10%是450，VoV的10%是800
    SAMPLE_OUTPUT = '/root/autodl-tmp/cyclegan/data/test03_filtered_uav_visloc/step2_sample_test'
    # 请根据 Step 1 跑出来的真实表格，修改下面这两个数值 👇
    # step2_sample_effect(INPUT_DIR, var_threshold=450.0, vov_threshold=800.0, sample_output_dir=SAMPLE_OUTPUT, num_samples=200)

    # 【执行阶段 3】：全量跑。
    # GOOD_OUTPUT = '/root/autodl-tmp/cyclegan/data/visloc_filtered_v2/trainA'
    # BAD_OUTPUT = '/root/autodl-tmp/cyclegan/data/visloc_filtered_v2/bad_trainA'
    # step3_full_filter(INPUT_DIR, GOOD_OUTPUT, BAD_OUTPUT, final_var_thresh=450.0, final_vov_thresh=800.0)