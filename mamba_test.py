import time
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# 尝试导入 selective_scan_fn 和 selective_scan_ref，如果无法导入会跳过
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm (in which causal_conv1d is needed)
# 如果 mamba_ssm 不可用，则尝试导入备用的 selective_scan 函数
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

# 修改 DropPath 的 `__repr__` 方法，以便更友好地显示 drop_prob 的值
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,  # 模型的输入特征维度
            d_state=16,  # 状态的大小
            d_conv=3,  # 卷积核大小
            expand=2,  # 特征维度的扩展因子
            dt_rank="auto",  # 时间步长张量的秩值
            dt_min=0.001,  # 动态时间步的下界
            dt_max=0.1,  # 动态时间步的上界
            dt_init="random",  # 动态时间权重的初始化方式
            dt_scale=1.0,  # 时间投影的缩放因子
            dt_init_floor=1e-4,  # 动态初始化时间步的低位阈值
            dropout=0.,  # Dropout 概率
            conv_bias=True,  # 卷积中的偏置
            bias=False,  # 投影层中的偏置
            device=None,  # 计算设备
            dtype=None,  # 数据类型
            **kwargs,  # 其他未定参数
    ):
        """
       SS2D（Selective Scan 2D）是一个复杂的模型组件，将卷积操作与自定义的选择性扫描（Selective Scan）机制结合在一起，主要用于特征的累积计算和交互式学习，并结合 LayerNorm 和 Dropout 控制模型训练。
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        # 基本模型参数
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv  # 卷积核大小
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)  # 内部的扩展维度  # 输入维度通过 expand 扩展后的输出
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank  # 定义时间秩，用于扫描机制  # 动态时间步长的秩

        # 定义输入与特征扩展的线性变换  # 输入特征的线性映射，扩展维度到两倍
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        # 深度卷积：`self.expand` 保证适配不同图片的大小和特征
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,  # 设置为组卷积，分割通道并在相同组内卷积
            bias=conv_bias,
            kernel_size=d_conv,  # 卷积核大小
            padding=(d_conv - 1) // 2,  # 保持输出特征图尺寸与输入一致
            **factory_kwargs,
        )  # Group卷积用于特征交互
        self.act = nn.SiLU()  # 激活函数

        # 高度复杂的特征变换与扫描权重初始化  # 四个 x_proj 线性层
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        # 合并四个投影层，构造可学习参数 (K, rank, dimension)
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        # 初始化动态时间投影函数，self.dt_projs 定义了时间步的动态特性，通过权重与偏置初始化来控制动态步长。
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        # 定义最终的归一化与输出
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        # dropout防止过拟合
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        # 初始化 dt 的投影层，保持权重和偏置在特定范围并满足初始化要求。
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):

        # 前向传播中的核心函数，利用选择性扫描机制从一定集群中提取特征。
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape  # 输入的批次通道与宽高
        L = H * W  # 特征的长度（平面的总像素数）
        K = 4  # 定义方向上的上下文个数（上下、左右方向）

        # 将特征图分解到 4 个方向（水平、垂直以及反方向）
        # x_hwwh - 提取两方向（水平和垂直）特征，转置后统一拼接到 (batch, 2, channels, length)
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l) # 拼接水平和垂直以及反向的多方向特征

        # 特征编码：通过 x_proj_weight 进行线性操作
        # 将多方向特征扩展后序列化处理 (b, k, d, l) -> (b, k, c, l)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        # 根据特征编码拆分结果：分别得到时间动态参数（dts）、特征权重（Bs）和重要性权重（Cs）
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        # 动态时间步扩展：通过时间步动态权重投影进行计算
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        # 将特征图序列化处理为二维张量以参与计算
        xs = xs.float().view(B, -1, L)  # (b, k * d, l) # 特征图序列化为 (b, k*d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l) # 时间步相关特性 (b, k*d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)  # 特征权重提取 (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)  # 重要性权重提取 (b, k, d_state, l)

        # 提取全局和局部信息的可学习参数
        Ds = self.Ds.float().view(-1)  # (k * d) # 全局参数 (k*d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state) # 局部参数，对应动态状态权重 (k*d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)  # 时间动态偏置 (k*d)

        # 使用选择性扫描函数计算多方向特征融合结果
        # 通过输入的动态时间步、权重等参数，返回融合后的多方向结果 (b, k, d, l)
        out_y = self.selective_scan(
            xs, dts,  # 输入序列化的多方向特征图和扩展的动态时间步
            As, Bs, Cs, Ds, z=None,  # 可学习权重参数和附加静态特征
            delta_bias=dt_projs_bias,  # 时间步偏置
            delta_softplus=True,  # 使用 softplus 激活函数
            return_last_state=False,  # 不返回最后状态
        ).view(B, K, -1, L)  # 重新调整输出大小到 (batch, 4方向特征, d_state, length)
        assert out_y.dtype == torch.float  # 确保输出为浮点类型

        # 将方向特征调整为 2 个反向特征，用于反向操作
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)  # 将两个方向特征反序列化
        # 重建水平和垂直特征，并恢复原来的批次和空间布局
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        # 返回所有结果包括：
        # - out_y[:, 0]: 多方向融合结果 (正向权重)
        # - inv_y[:, 0]: 多方向反向结果 (反向权重)
        # - wh_y: 水平特征
        # - invwh_y: 垂直特征

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        # 这是 forward_corev0 的替代版本，逻辑基本一致，仅切换使用了不同版本的选择性扫描函数。
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        """
        完整的前向传播逻辑。
        """
        B, C, H, W = x.shape  # 获取输入形状
        print(f"Step 0 (Input): {x.shape}")

        # Step 1: in_proj (转换输入)
        xz = self.in_proj(x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C))  # 调整维度为 [B, H*W, C]
        print(f"Step 1 (in_proj output): {xz.shape}")

        x, z = xz.chunk(2, dim=-1)  # 分割嵌入为两部分
        print(f"Step 2 (x chunk): {x.shape}, z chunk: {z.shape}")

        x = x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # 恢复 [B, C, H, W]
        print(f"Step 3 (x reshaped): {x.shape}")

        # 卷积提取局部特征
        x = self.act(self.conv2d(x))
        print(f"Step 4 (convolution output): {x.shape}")

        # 核心特征提取
        y1, y2, y3, y4 = self.forward_core(x)
        print(f"Step 5 (y1): {y1.shape}, (y2): {y2.shape}, (y3): {y3.shape}, (y4): {y4.shape}")

        # 综合方向特性
        y = y1 + y2 + y3 + y4
        print(f"Step 6 (y combined): {y.shape}")

        # 更新通道数并调整形状
        B, C_new, L = y.shape  # 当前形状 [Batch, Channel, Flattened spatial size]
        H = W = int(L ** 0.5)  # 假设特征大小为方形，计算高度和宽度
        y = y.view(B, C_new, H, W)
        print(f"Step 7 (y reshaped to [B, C, H, W]): {y.shape}")

        # 输出归一化
        y = self.out_norm(y.permute(0, 2, 3, 1))  # 转换为 NHWC 格式
        y = y * F.silu(z.view(B, H, W, -1))
        y = y.permute(0, 3, 1, 2).contiguous()  # 恢复为 [B, C, H, W]
        print(f"Step 8 (final y permuted): {y.shape}")

        # 在送入 self.out_proj 前，正确展平
        y = y.permute(0, 2, 3, 1).contiguous().view(B * H * W, C_new)  # [Flattened spatial size, Channels]
        print(f"Step 8.1 (y reshaped for Linear): {y.shape}")

        # 调整输出维度
        out = self.out_proj(y)  # [B * H * W, d_model]
        print(f"Step 9 (out_proj output): {out.shape}")

        # 恢复原始尺寸
        out = out.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # 恢复为 [B, d_model, H, W]
        print(f"Step 10 (Final output reshaped): {out.shape}")

        if self.dropout is not None:
            out = self.dropout(out)  # Dropout

        return out


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = nn.BatchNorm2d,  # 修改为适用于图像的归一化
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs,
    ):
        """
        VSSBlock 是一个单个块，包含归一化、自注意力和 DropPath 机制。
       Args:
           hidden_dim: 输入特征的维度(通道数)。
           drop_path: DropPath 的概率，用于随机丢弃路径。
           norm_layer: 使用的归一化层（默认为 LayerNorm）。
           attn_drop_rate: 注意力的 Dropout 率。
           d_state: 模型中的状态维度（通常在自注意力中使用）。
       """

        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)  # 直接使用 BatchNorm2d 或 GroupNorm
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)  # 自定义 SS2D 模块
        self.drop_path = DropPath(drop_path)  # DropPath 随机丢弃路径模块

    def forward(self, input: torch.Tensor):
        """
       前向传播：
       - 对输入进行归一化。
       - 使用 SS2D 模块计算自注意力。
       - 应用 DropPath 将结果加回到输入中。
       """
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))  # 加上 Residual 连接
        return x


if __name__ == "__main__":
    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 输入张量
    input = torch.randn(1, 16, 128, 128).to(device)  # 假设 Batch=1, Channels=16, 高=宽=128

    # 初始化 VSSBlock
    vss_block = VSSBlock(hidden_dim=16, drop_path=0.1, attn_drop_rate=0.1, d_state=16).to(device)

    # 前向传播
    output = vss_block(input)

    print("Input shape:", input.shape)
    print("Output shape:", output.shape)  # 验证 VSSBlock 的输出形状是否正确

# if __name__ == "__main__":
#     # 初始化设备
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Running on device: {device}")

#     # 输入张量
#     input = torch.randn(1, 16, 128, 128).to(device)  # 假设 Batch=1, Channels=16, Height=128, Width=128

#     # 初始化 SS2D
#     ss2d = SS2D(d_model=16, dropout=0, d_state=16).to(device)

#     # 前向传播
#     output = ss2d(input)

#     print("Input shape:", input.shape)
#     print("Output shape:", output.shape)  # 验证输出是否符合预期


