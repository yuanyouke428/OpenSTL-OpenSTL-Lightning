import re
import matplotlib.pyplot as plt
import os
import pandas as pd  # 引入 pandas 用于保存 csv 数据

# 1. 配置区域
log_file = '/home/ps/data2/zp/OpenSTL-OpenSTL-Lightning/work_dirs/CIKM_SimVP_Aligned/train_20251224_162739.log'
save_dir = './vis_results_loss'  # 指定保存目录
img_name = 'loss_curve.png'
data_name = 'training_log_data.csv'

# 确保保存目录存在
os.makedirs(save_dir, exist_ok=True)

epochs = []
train_losses = []
val_losses = []

# 2. 读取并解析日志
print(f"正在读取日志: {log_file} ...")
try:
    with open(log_file, 'r') as f:
        content = f.read()

    blocks = re.split(r'(?=Epoch \d+:)', content)

    for block in blocks:
        if "Epoch" not in block:
            continue

        epoch_match = re.search(r'Epoch\s+(\d+)', block)
        train_match = re.search(r'Train Loss:\s+([\d\.]+)', block)
        val_match = re.search(r'Vali Loss:\s+([\d\.]+)', block)

        if epoch_match and train_match and val_match:
            epochs.append(int(epoch_match.group(1)))
            train_losses.append(float(train_match.group(1)))
            val_losses.append(float(val_match.group(1)))

    print(f"解析完成，共 {len(epochs)} 个 Epoch。")

    # 3. 绘制并保存曲线图
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', color='#1f77b4', linewidth=2)
    plt.plot(epochs, val_losses, label='Validation Loss', color='#ff7f0e', linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # --- 保存图像逻辑 ---
    img_path = os.path.join(save_dir, img_name)
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭画布，释放内存
    print(f"✅ 曲线图已保存: {img_path}")

    # 4. 保存解析后的数据到 CSV (新增逻辑)
    # 这样下次你可以直接看数据，不用再跑正则解析了
    data_path = os.path.join(save_dir, data_name)
    df = pd.DataFrame({
        'Epoch': epochs,
        'Train Loss': train_losses,
        'Vali Loss': val_losses
    })
    df.to_csv(data_path, index=False)
    print(f"✅ 数据表已保存: {data_path}")

except FileNotFoundError:
    print(f"❌ 错误: 找不到文件 {log_file}")
except Exception as e:
    print(f"❌ 发生未知错误: {e}")