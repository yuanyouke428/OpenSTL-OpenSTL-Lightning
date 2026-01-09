import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--raw_root', type=str, default='./data/sevir')
args = parser.parse_args()

print(f"🕵️  正在检查根目录: {os.path.abspath(args.raw_root)}")

# 1. 检查 CATALOG
catalog_path = os.path.join(args.raw_root, 'CATALOG.csv')
if not os.path.exists(catalog_path):
    print(f"❌ 错误: 在 {args.raw_root} 没找到 CATALOG.csv")
    # 尝试退一级
    parent = os.path.dirname(args.raw_root)
    catalog_path = os.path.join(parent, 'CATALOG.csv')
    if os.path.exists(catalog_path):
        print(f"✅ 但是在上一级找到了: {catalog_path}")
    else:
        print("   请确认 CATALOG.csv 的位置。")
        exit()
else:
    print(f"✅ 找到 CATALOG.csv")

# 2. 读取 CSV 看它想要什么文件
df = pd.read_csv(catalog_path, nrows=100)
vil_files = df[df['img_type'] == 'vil']['file_name'].unique()
print(f"\n📋 CATALOG.csv 里列出的文件示例 (前3个):")
for f in vil_files[:15]:
    print(f"   - {f}")

# 3. 搜索硬盘上的文件
print(f"\n🔍 正在扫描硬盘上的 .h5 文件...")
found_files = []
for root, dirs, files in os.walk(args.raw_root):
    for file in files:
        if file.endswith(".h5"):
            found_files.append(os.path.join(root, file))

if len(found_files) == 0:
    print(f"❌ 在 {args.raw_root} 及其子目录下没有找到任何 .h5 文件！")
    print("   -> 请检查路径参数 --raw_root 是否正确。")
else:
    print(f"✅ 找到了 {len(found_files)} 个 .h5 文件。路径示例:")
    for f in found_files[:3]:
        print(f"   - {f}")

    # 4. 匹配测试
    print(f"\n🧪 匹配测试:")
    target_file = os.path.basename(vil_files[0])  # 取 CSV 里第一个文件名
    print(f"   目标文件: {target_file}")

    match = False
    for local_f in found_files:
        if target_file in local_f:
            print(f"   ✅ 成功匹配到: {local_f}")
            match = True
            break

    if not match:
        print(f"   ❌ 无法匹配！虽然硬盘有文件，但名字好像对不上，或者层级太深。")