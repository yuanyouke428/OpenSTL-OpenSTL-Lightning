import torch
import torch.nn as nn
import time

# 尝试导入 Mamba，如果报错说明环境没装好
try:
    from mamba_ssm import Mamba

    print("✅ Mamba-ssm 导入成功！")
except ImportError:
    print("❌ 错误：找不到 mamba_ssm，请检查 pip install 是否成功。")
    exit()


class SimpleMambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        # Mamba 官方模块
        self.mamba = Mamba(
            d_model=dim,  # 特征维度
            d_state=d_state,  # 内部状态维度
            d_conv=d_conv,  # 局部卷积宽度
            expand=expand,  # 扩展因子
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x 的形状是图片: [Batch, Channel, Height, Width]
        B, C, H, W = x.shape
        print(f"1. 输入形状: {x.shape}")

        # --- 关键步骤：图片转序列 ---
        # 变成 [Batch, H*W, Channel] -> Mamba 需要 (B, L, D)
        x_flat = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        print(f"2. 展平后形状 (符合Mamba输入): {x_flat.shape}")

        # 归一化
        x_norm = self.norm(x_flat)

        # Mamba 推理
        out = self.mamba(x_norm)
        print(f"3. Mamba输出形状: {out.shape}")

        # --- 关键步骤：序列转回图片 ---
        # 变回 [Batch, Channel, Height, Width]
        out = out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        print(f"4. 恢复图片形状: {out.shape}")

        return out + x  # 残差连接


# --- 开始测试 ---
def run_demo():
    # 检查是否有 GPU，Mamba 必须在 CUDA 上跑（除非使用最新版的 CPU 兼容模式，但很慢）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 运行设备: {device}")

    if device == "cpu":
        print("⚠️ 警告：Mamba 在 CPU 上运行可能会报错或极慢，建议使用 GPU。")

    # 1. 定义模型 (假设特征通道是 64)
    model = SimpleMambaLayer(dim=64).to(device)

    # 2. 造一个假数据 (模拟 Batch=2, Channel=64, 64x64 的雷达图)
    x = torch.randn(2, 64, 64, 64).to(device)

    # 3. 运行前向传播
    start_time = time.time()
    try:
        y = model(x)
        end_time = time.time()
        print(f"\n✅ 测试通过！前向传播耗时: {end_time - start_time:.4f} 秒")
        print(f"最终输出张量形状: {y.shape}")
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        print("💡 提示：如果是 CUDA 相关错误，请检查 PyTorch 和 CUDA 版本是否匹配。")


if __name__ == "__main__":
    run_demo()