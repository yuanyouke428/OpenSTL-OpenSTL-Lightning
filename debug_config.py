import sys
import os
from mmengine.config import Config

# 指定你的配置文件路径
config_path = 'configs/cikm/simvp.py'

print(f"尝试加载: {config_path}")
if not os.path.exists(config_path):
    print("错误: 文件不存在！请检查路径")
else:
    try:
        cfg = Config.fromfile(config_path)
        print("成功加载配置！内容如下：")
        print(cfg)
    except Exception as e:
        print("\n!!!!!!!!!!!!!! 配置加载失败 !!!!!!!!!!!!!!")
        print("具体的错误信息如下（请把这个发给我）：")
        print(e)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")