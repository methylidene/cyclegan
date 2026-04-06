import cv2
import os
import shutil
import numpy as np
import random
from tqdm import tqdm

# ================= 核心算法 =================
def calculate_blur_variance(image_path):
    """计算图像的拉普拉斯方差，值越小代表图像越平坦/纯色"""
    image = cv2.imread(image_path)
    if image is None:
        return -1 # 读取失败的损坏图片
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# ================= 第一步：试算分布 (探底细) =================
def step1_analyze_distribution(source_dir, sample_size=1000):
    """随机抽取一批图片，计算方差分布，为你提供阈值参考"""
    images = [f for f in os.listdir(source_dir) if f.endswith(('.png', '.jpg','.JPG'))]
    sample_images = random.sample(images, min(sample_size, len(images)))
    
    print(f"\n🔍 [Step 1] 正在对 {len(sample_images)} 张图片进行方差分布试算...")
    variances = []
    for img_name in tqdm(sample_images):
        var = calculate_blur_variance(os.path.join(source_dir, img_name))
        if var != -1:
            variances.append(var)
            
    variances = np.array(variances)
    print("\n📊 === 数据集拉普拉斯方差分布报告 ===")
    print(f"最平滑 (Min): {np.min(variances):.2f} (大概率是纯海面/纯森林)")
    print(f"10% 分位数:   {np.percentile(variances, 10):.2f} (极度可疑的平滑区域)")
    print(f"25% 分位数:   {np.percentile(variances, 25):.2f}")
    print(f"中位数 (Med): {np.median(variances):.2f}")
    print(f"平均值 (Avg): {np.mean(variances):.2f}")
    print(f"最丰富 (Max): {np.max(variances):.2f} (大概率是密集的城市建筑群)")
    print("=====================================\n")
    return variances

# ================= 第二步：抽样盲测 (试效果) =================
def step2_sample_effect(source_dir, threshold, sample_output_dir, num_samples=100):
    """根据你给定的阈值，随机抓取一些图片分类，让你肉眼确认是否误杀"""
    good_sample_dir = os.path.join(sample_output_dir, 'sample_good')
    bad_sample_dir = os.path.join(sample_output_dir, 'sample_bad')
    
    # 每次试算前清空之前的抽样文件夹
    if os.path.exists(sample_output_dir):
        shutil.rmtree(sample_output_dir)
    os.makedirs(good_sample_dir, exist_ok=True)
    os.makedirs(bad_sample_dir, exist_ok=True)

    images = [f for f in os.listdir(source_dir) if f.endswith(('.png', '.jpg'))]
    sample_images = random.sample(images, min(num_samples, len(images)))
    
    print(f"\n🧪 [Step 2] 正在使用阈值 {threshold} 进行抽样测试 (抽取 {len(sample_images)} 张)...")
    bad_count = 0
    for img_name in tqdm(sample_images):
        img_path = os.path.join(source_dir, img_name)
        var = calculate_blur_variance(img_path)
        
        # 将方差值写在文件名前面，方便你肉眼审查时直接看到具体数值！
        new_name = f"var_{var:.1f}_{img_name}" 
        
        if var < threshold:
            shutil.copy(img_path, os.path.join(bad_sample_dir, new_name))
            bad_count += 1
        else:
            shutil.copy(img_path, os.path.join(good_sample_dir, new_name))
            
    print(f"✅ 抽样完成！在 {len(sample_images)} 张图中，拦截了 {bad_count} 张废片 (占比 {bad_count/len(sample_images)*100:.1f}%)。")
    print(f"👉 请立刻打开 VS Code 左侧的资源管理器，查看 {sample_output_dir} 里的结果！\n")

# ================= 第三步：全量过滤 (动真格) =================
def step3_full_filter(source_dir, good_dir, bad_dir, final_threshold):
    """确认阈值无误后，对全量数据集进行终极清洗"""
    os.makedirs(good_dir, exist_ok=True)
    os.makedirs(bad_dir, exist_ok=True)

    images = [f for f in os.listdir(source_dir) if f.endswith(('.png', '.jpg'))]
    
    print(f"\n🚀 [Step 3] 正在使用最终阈值 {final_threshold} 清洗全量数据 ({len(images)} 张)...")
    for img_name in tqdm(images):
        img_path = os.path.join(source_dir, img_name)
        var = calculate_blur_variance(img_path)
        
        if var < final_threshold:
            shutil.copy(img_path, os.path.join(bad_dir, img_name))
        else:
            shutil.copy(img_path, os.path.join(good_dir, img_name))
    print(f"🎉 全量清洗完成！极其纯净的训练数据已存入: {good_dir}\n")


if __name__ == '__main__':
    # 你的无人机数据集路径 (以 A 域为例)
    INPUT_DIR = '/root/autodl-tmp/cyclegan/data/test02/trainA'
    
    # ---------------- 流程控制台 ----------------
    # 建议你每次只解除其中一个注释，分步执行！

    # 【执行阶段 1】：探底细。看看你的数据方差分布大概在什么范围
    step1_analyze_distribution(INPUT_DIR, sample_size=2000)

    # 【执行阶段 2】：试效果。结合第一步打印的 10% 分位数，填入你猜的阈值
    # SAMPLE_OUTPUT = '/root/autodl-tmp/cyclegan/data/step2_sample_test'
    # step2_sample_effect(INPUT_DIR, threshold=400.0, sample_output_dir=SAMPLE_OUTPUT, num_samples=200)

    # 【执行阶段 3】：全量跑。当你通过第二步，肉眼确认 bad 文件夹里确实全是纯色废片，而 good 文件夹里没有误杀好图时，执行这步！
    # GOOD_OUTPUT = '/root/autodl-tmp/cyclegan/data/visloc_filtered/trainA'
    # BAD_OUTPUT = '/root/autodl-tmp/cyclegan/data/visloc_filtered/bad_trainA'
    # step3_full_filter(INPUT_DIR, GOOD_OUTPUT, BAD_OUTPUT, final_threshold=400.0)