# # # import os
# # # import sys
# # #
# # # import pywt
# # # from mmengine.model import BaseModule
# # # from mmdet.registry import MODELS
# # # from functools import partial
# # # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../condinstmam/model")))
# # # import torch
# # # import torch.nn as nn
# # # import torch.nn.functional as F
# # # from functools import partial
# # # from models.vmamba import Backbone_VSSM
# # # from models.vmamba import SS2D
# # # @MODELS.register_module()
# # # class MM_VSSM(BaseModule, Backbone_VSSM):
# # #     def __init__(self, *args, **kwargs):
# # #         BaseModule.__init__(self)
# # #         Backbone_VSSM.__init__(self, *args, **kwargs)
# # #
# # #
# # # @MODELS.register_module()
# # # class ASS2D(SS2D):
# # #     def __init__(self, *args, **kwargs):
# # #         SS2D.__init__(self, *args, **kwargs)
# # #
# # #
# # # def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
# # #     w = pywt.Wavelet(wave)
# # #     dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
# # #     dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
# # #     dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
# # #                                dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
# # #                                dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
# # #                                dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)
# # #
# # #     dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)
# # #
# # #     rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
# # #     rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
# # #     rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
# # #                                rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
# # #                                rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
# # #                                rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)
# # #
# # #     rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)
# # #
# # #     return dec_filters, rec_filters
# # #
# # # def wavelet_transform(x, filters):
# # #     b, c, h, w = x.shape
# # #     pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
# # #     x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
# # #     x = x.reshape(b, c, 4, h // 2, w // 2)
# # #     return x
# # #
# # #
# # # def inverse_wavelet_transform(x, filters):
# # #     b, c, _, h_half, w_half = x.shape
# # #     pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
# # #     x = x.reshape(b, c * 4, h_half, w_half)
# # #     x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
# # #     return x
# # #
# # # # class MBWTConv2d(nn.Module):
# # # #     def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1',ssm_ratio=1,forward_type="v05",):
# # # #         super(MBWTConv2d, self).__init__()
# # # #
# # # #         self.in_channels = in_channels
# # # #         self.wt_levels = wt_levels
# # # #         self.stride = stride
# # # #         self.dilation = 1
# # # #
# # # #         self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
# # # #         self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
# # # #         self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
# # # #
# # # #         self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
# # # #         self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)
# # # #
# # # #         self.global_atten =ASS2D(d_model=in_channels, d_state=1,
# # # #              ssm_ratio=ssm_ratio, initialize="v2", forward_type=forward_type, channel_first=True, k_group=2)
# # # #         self.base_scale = _ScaleModule([1, in_channels, 1, 1])
# # # #
# # # #         self.wavelet_convs = nn.ModuleList(
# # # #             [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
# # # #                        groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
# # # #         )
# # # #
# # # #         self.wavelet_scale = nn.ModuleList(
# # # #             [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
# # # #         )
# # # #
# # # #         if self.stride > 1:
# # # #             self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
# # # #             self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
# # # #                                                    groups=in_channels)
# # # #         else:
# # # #             self.do_stride = None
# # # #
# # # #         # 可学习门控：融合 x 与 x_tag，用 sigmoid 控制比例
# # # #         self.fusion_gate = nn.Sequential(
# # # #             nn.Conv2d(in_channels, in_channels, kernel_size=1),
# # # #             nn.Sigmoid()
# # # #         )
# # # #
# # # #     def forward(self, x):
# # # #
# # # #         x_ll_in_levels = []
# # # #         x_h_in_levels = []
# # # #         shapes_in_levels = []
# # # #
# # # #         curr_x_ll = x
# # # #
# # # #         for i in range(self.wt_levels):
# # # #             curr_shape = curr_x_ll.shape
# # # #             shapes_in_levels.append(curr_shape)
# # # #             if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
# # # #                 curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
# # # #                 curr_x_ll = F.pad(curr_x_ll, curr_pads)
# # # #
# # # #             curr_x = self.wt_function(curr_x_ll)
# # # #             curr_x_ll = curr_x[:, :, 0, :, :]
# # # #
# # # #             shape_x = curr_x.shape
# # # #             curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
# # # #             curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
# # # #             curr_x_tag = curr_x_tag.reshape(shape_x)
# # # #
# # # #             x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
# # # #             x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])
# # # #
# # # #         next_x_ll = 0
# # # #
# # # #         for i in range(self.wt_levels - 1, -1, -1):
# # # #             curr_x_ll = x_ll_in_levels.pop()
# # # #             curr_x_h = x_h_in_levels.pop()
# # # #             curr_shape = shapes_in_levels.pop()
# # # #
# # # #             curr_x_ll = curr_x_ll + next_x_ll
# # # #
# # # #             curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
# # # #             next_x_ll = self.iwt_function(curr_x)
# # # #
# # # #             next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]
# # # #
# # # #         x_tag = next_x_ll
# # # #         assert len(x_ll_in_levels) == 0
# # # #
# # # #         x = self.base_scale(self.global_atten(x))
# # # #         alpha = self.fusion_gate(x_tag)
# # # #         x = x * (1 - alpha) + x_tag * alpha
# # # #
# # # #         if self.do_stride is not None:
# # # #             x = self.do_stride(x)
# # # #
# # # #         return x
# # #
# # #
# # #
# # # # 假设下面两个工具函数已在 vmamba 或你自己的库中定义：
# # # # create_wavelet_filter(wt_type, in_ch, out_ch, dtype)
# # # # wavelet_transform(x, filters)
# # # # inverse_wavelet_transform(x, filters)
# # #
# # # class MBWTConv2d(nn.Module):
# # #     def __init__(
# # #         self,
# # #         in_channels,
# # #         out_channels,
# # #         kernel_size=5,
# # #         stride=1,
# # #         bias=True,
# # #         wt_levels=1,
# # #         wt_type='db1',
# # #         ssm_ratio=1,
# # #         forward_type="v05",
# # #     ):
# # #         super(MBWTConv2d, self).__init__()
# # #
# # #         # 不再强制 in_channels == out_channels
# # #         self.in_channels = in_channels
# # #         self.out_channels = out_channels
# # #         self.wt_levels = wt_levels
# # #         self.stride = stride
# # #         self.dilation = 1
# # #
# # #         # 构建小波变换和逆变换的滤波器，并固定（requires_grad=False）
# # #         self.wt_filter, self.iwt_filter = create_wavelet_filter(
# # #             wt_type, in_channels, in_channels, torch.float
# # #         )
# # #         self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
# # #         self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
# # #
# # #         # 用于快速调用的小波函数
# # #         self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
# # #         self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)
# # #
# # #         # 全局状态空间建模：输入通道是 in_channels，输出也是 in_channels
# # #         self.global_atten = ASS2D(
# # #             d_model=in_channels,
# # #             d_state=1,
# # #             ssm_ratio=ssm_ratio,
# # #             initialize="v2",
# # #             forward_type=forward_type,
# # #             channel_first=True,
# # #             k_group=2
# # #         )
# # #         # 基础尺度变换也是针对 in_channels
# # #         self.base_scale = _ScaleModule([1, in_channels, 1, 1])
# # #
# # #         # 小波域 depthwise 卷积：每层都对 in_channels * 4 做 depthwise
# # #         self.wavelet_convs = nn.ModuleList([
# # #             nn.Conv2d(
# # #                 in_channels * 4,
# # #                 in_channels * 4,
# # #                 kernel_size,
# # #                 padding='same',
# # #                 stride=1,
# # #                 dilation=1,
# # #                 groups=in_channels * 4,
# # #                 bias=False
# # #             )
# # #             for _ in range(self.wt_levels)
# # #         ])
# # #         # 对应的小波尺度缩放（可学习）
# # #         self.wavelet_scale = nn.ModuleList([
# # #             _ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1)
# # #             for _ in range(self.wt_levels)
# # #         ])
# # #
# # #         # 如果需要下采样，则在融合后通过 depthwise conv 做 stride
# # #         if self.stride > 1:
# # #             # 注意：这里要对 out_channels 做 depthwise
# # #             # 因此 weight 尺寸是 [out_channels, 1, 1, 1]，groups=out_channels
# # #             self.stride_filter = nn.Parameter(
# # #                 torch.ones(out_channels, 1, 1, 1),
# # #                 requires_grad=False
# # #             )
# # #             self.do_stride = lambda x_in: F.conv2d(
# # #                 x_in,
# # #                 self.stride_filter,
# # #                 bias=None,
# # #                 stride=self.stride,
# # #                 groups=out_channels
# # #             )
# # #         else:
# # #             self.do_stride = None
# # #
# # #         # 最后，需要把 in_channels 投影到 out_channels
# # #         # 这样无论 in/out 是否相等，都会得到正确的通道数
# # #         self.proj = nn.Conv2d(
# # #             in_channels, out_channels,
# # #             kernel_size=1,
# # #             stride=1,
# # #             padding=0,
# # #             bias=bias
# # #         )
# # #
# # #         # 可学习门控：融合 x 与 x_tag，用 sigmoid 控制比例
# # #         self.fusion_gate = nn.Sequential(
# # #             nn.Conv2d(in_channels, in_channels, kernel_size=1),
# # #             nn.Sigmoid()
# # #         )
# # #
# # #     def forward(self, x):
# # #         """
# # #         x: [B, in_channels, H, W]
# # #         返回: [B, out_channels, H//stride, W//stride]
# # #         """
# # #         # =========================
# # #         # 1. 小波分解 + 处理
# # #         # =========================
# # #         x_ll_in_levels = []
# # #         x_h_in_levels = []
# # #         shapes_in_levels = []
# # #
# # #         curr_x_ll = x  # 初始 LL 分支是输入
# # #
# # #         for i in range(self.wt_levels):
# # #             curr_shape = curr_x_ll.shape  # [B, C, H, W]
# # #             shapes_in_levels.append(curr_shape)
# # #
# # #             # 如果 H 或 W 是奇数，通过 pad 使得能被 2 整除
# # #             if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
# # #                 curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
# # #                 curr_x_ll = F.pad(curr_x_ll, curr_pads)
# # #
# # #             # 执行小波正向变换：输出尺寸 [B, C, 4, H//2, W//2]
# # #             curr_x = self.wt_function(curr_x_ll)
# # #             # 提取 LL 分量作为下一层的小波输入
# # #             curr_x_ll = curr_x[:, :, 0, :, :]
# # #
# # #             # 把 4 个子带 reshape 到 “通道堆叠” 形式，方便做 depthwise 卷积
# # #             shape_x = curr_x.shape  # [B, C, 4, H', W']
# # #             # reshape → [B, C*4, H', W']
# # #             curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
# # #
# # #             # depthwise 卷积 + 尺度缩放
# # #             curr_x_tag = self.wavelet_scale[i](
# # #                 self.wavelet_convs[i](curr_x_tag)
# # #             )
# # #             # 再 reshape 回 [B, C, 4, H', W']
# # #             curr_x_tag = curr_x_tag.reshape(shape_x)
# # #
# # #             # 保存本层：LL 分量和高频分量（LH/HL/HH）
# # #             x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])    # [B, C, H', W']
# # #             x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])   # [B, C, 3, H', W']
# # #
# # #         # =========================
# # #         # 2. 小波逆变换——自底向上重建
# # #         # =========================
# # #         next_x_ll = 0
# # #         # 从最后一层往前逐层逆变换
# # #         for i in range(self.wt_levels - 1, -1, -1):
# # #             curr_x_ll = x_ll_in_levels.pop()   # 上一层的 LL
# # #             curr_x_h = x_h_in_levels.pop()     # 对应的高频
# # #             curr_shape = shapes_in_levels.pop()
# # #
# # #             # 跨层残差：把上一层的重建（next_x_ll）加到当前层 LL
# # #             curr_x_ll = curr_x_ll + next_x_ll
# # #
# # #             # 拼回 [LL, LH, HL, HH]
# # #             curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
# # #             # 逆小波
# # #             next_x_ll = self.iwt_function(curr_x)
# # #             # 有可能尺寸被 pad，需要裁剪回原始 H×W
# # #             next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]
# # #
# # #         # next_x_ll: [B, in_channels, H, W] —— 这是小波分支输出
# # #         x_tag = next_x_ll
# # #         assert len(x_ll_in_levels) == 0
# # #
# # #         # =========================
# # #         # 3. 全局状态空间建模 & 基础尺度缩放
# # #         # =========================
# # #         # global_atten(x) 的输出尺寸是 [B, in_channels, H, W]
# # #         x = self.base_scale(self.global_atten(x))  # 先全局建模，再尺度调节
# # #
# # #         # 和小波分支做残差融合
# # #         alpha = self.fusion_gate(x_tag)
# # #         x = x * (1 - alpha) + x_tag * alpha
# # #
# # #         # =========================
# # #         # 4. 投影到 out_channels
# # #         # =========================
# # #         x = self.proj(x)  # [B, out_channels, H, W]
# # #
# # #         # =========================
# # #         # 5. 如果需要下采样，则在投影后对 out_channels 做 depthwise stride
# # #         # =========================
# # #         if self.do_stride is not None:
# # #             x = self.do_stride(x)  # [B, out_channels, H//stride, W//stride]
# # #
# # #         return x
# # #
# # #
# # #
# # # class _ScaleModule(nn.Module):
# # #     def __init__(self, dims, init_scale=1.0, init_bias=0):
# # #         super(_ScaleModule, self).__init__()
# # #         self.dims = dims
# # #         self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
# # #         self.bias = None
# # #
# # #     def forward(self, x):
# # #         return torch.mul(self.weight, x)
# # #
# # # if __name__ == '__main__':
# # #     x = torch.rand(1, 8, 64, 64).cuda()
# # #     m = MBWTConv2d(8, 169, 3).cuda()
# # #     y = m(x)
# # #     # y = d(x)
# # #     print(y.shape)
# #
# #
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# #
# # class AdaDConvSimple(nn.Module):
# #     """
# #     简化版的自适应加权下采样（Adaptive-weighted Downsampling）模块，
# #     支持自定义输出通道数 out_channels。相比原版做了如下优化：
# #       1. 去掉分组和 NIN 支持，仅保留单组动态权重计算；
# #       2. 增加 1x1 卷积，用于把中间通道映射到 out_channels；
# #       3. 可选通道注意力（SE-like），默认为关闭以节省开销；
# #       4. 计算流程更紧凑：unfold + reshape + softmax + weighted sum。
# #     """
# #
# #     def __init__(self,
# #                  in_channels: int,
# #                  out_channels: int,
# #                  kernel_size: int = 3,
# #                  stride: int = 1,
# #                  use_channel: bool = False):
# #         """
# #         Args:
# #             in_channels (int): 输入特征图通道数。
# #             out_channels (int): 输出特征图通道数。
# #             kernel_size (int): 卷积核大小（默认 3）。
# #             stride (int): 下采样步长（默认 1，即不降采样；若要降采样可设为 2）。
# #             use_channel (bool): 是否使用通道注意力（SE-like）；若 False，则不额外开销。
# #         """
# #         super().__init__()
# #         self.in_channels = in_channels
# #         self.out_channels = out_channels
# #         self.kernel_size = kernel_size
# #         self.stride = stride
# #         self.padding = (kernel_size - 1) // 2
# #
# #         # —— 1. 生成空间动态权重的网络 ——
# #         # 对输入 x → conv(kernel_size, stride) → BN → 得到 (B, in_channels * K*K, H', W')
# #         self.weight_net = nn.Sequential(
# #             nn.Conv2d(in_channels=in_channels,
# #                       out_channels=in_channels * (kernel_size**2),
# #                       kernel_size=kernel_size,
# #                       stride=stride,
# #                       padding=self.padding,
# #                       bias=False),
# #             nn.BatchNorm2d(in_channels * (kernel_size**2))
# #         )
# #
# #
# #         # —— 3. 用 1x1 卷积把中间 in_channels 映射到 out_channels ——
# #         #    这样就能自定义输出通道数
# #         self.project = nn.Conv2d(in_channels,
# #                                  out_channels,
# #                                  kernel_size=1,
# #                                  bias=False)
# #
# #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# #         """
# #         Args:
# #             x: 输入特征图，shape = [B, in_channels, H, W]
# #         Returns:
# #             out: 输出特征图，shape = [B, out_channels, H', W']
# #         """
# #         b, c, h, w = x.shape
# #
# #         # —— 1. 计算输出特征图的高宽 ——
# #         h_out = (h + 2 * self.padding - (self.kernel_size - 1) - 1) // self.stride + 1
# #         w_out = (w + 2 * self.padding - (self.kernel_size - 1) - 1) // self.stride + 1
# #
# #         # —— 2. 生成空间动态权重 ——
# #         # weight_raw: [B, in_channels * K*K, H', W']
# #         weight_raw = self.weight_net(x)
# #         # reshape -> [B, in_channels, K*K, H', W']
# #         weight = weight_raw.view(b, c, self.kernel_size * self.kernel_size, h_out, w_out)
# #
# #         # —— 4. 对 kernel_size*K*K 维度做 softmax ——
# #         #    保证动态卷积核在空间邻域上的加权和为 1
# #         weight = weight.softmax(dim=2)  # [B, in_channels, K*K, H', W']
# #
# #         # —— 5. 提取滑动窗口补丁 → unfold → reshape ——
# #         # pad reflect 以便边缘也能取到邻域
# #         x_pad = F.pad(x, pad=[self.padding]*4, mode='reflect')
# #         # unfold 获得形状 [B, in_channels, H'*K, W'*K] 其中最后两维是把 K*K 抽平后的
# #         patches = x_pad.unfold(2, self.kernel_size, self.stride) \
# #                        .unfold(3, self.kernel_size, self.stride)
# #         # 此时 patches.shape = [B, in_channels, H', W', K, K]
# #         patches = patches.contiguous() \
# #                          .view(b, c, h_out, w_out, self.kernel_size * self.kernel_size)
# #         # 再把最后一维挪到 dim=2 以便和 weight 对齐
# #         patches = patches.permute(0, 1, 4, 2, 3)  # [B, in_channels, K*K, H', W']
# #
# #         # —— 6. 加权求和得到中间特征 ——
# #         # 对应每个通道，在 H',W' 处，用 weight 对邻域像素做加权
# #         # patches 和 weight 都是 [B, in_channels, K*K, H', W']
# #         out_mid = (weight * patches).sum(dim=2)  # [B, in_channels, H', W']
# #
# #         # —— 7. 1×1 卷积投射到目标通道数 out_channels ——
# #         out = self.project(out_mid)  # [B, out_channels, H', W']
# #         print(out.shape)
# #         return out
# #
# #
# # if __name__ == '__main__':
# #     # 假设输入特征 x: [B, 64, 128, 128]
# #     model = AdaDConvSimple(
# #         in_channels=64,
# #         out_channels=128,  # 想要输出 128 通道
# #         kernel_size=3,
# #         stride=2,  # 下采样到 (128→64)
# #     )
# #
# #     x = torch.randn(2, 64, 128, 128)
# #     y = model(x)  # y.shape == [2, 128, 64, 64]
# #     print(y.shape)
#
#
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# #
# # class wConv2d(nn.Module):
# #     def __init__(self, in_channels, out_channels, kernel_size, den, stride=1, padding=1, groups=1, bias=False):
# #         super(wConv2d, self).__init__()
# #         self.stride = stride
# #         self.padding = padding
# #         self.kernel_size = kernel_size
# #         self.groups = groups
# #         self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
# #         nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
# #         self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
# #
# #         device = torch.device('cpu')
# #         self.register_buffer('alfa', torch.cat([torch.tensor(den, device=device),torch.tensor([1.0], device=device),torch.flip(torch.tensor(den, device=device), dims=[0])]))
# #         self.register_buffer('Phi', torch.outer(self.alfa, self.alfa))
# #
# #         if self.Phi.shape != (kernel_size, kernel_size):
# #             raise ValueError(f"Phi shape {self.Phi.shape} must match kernel size ({kernel_size}, {kernel_size})")
# #
# #     def forward(self, x):
# #         Phi = self.Phi.to(x.device)
# #         weight_Phi = self.weight * Phi
# #         return F.conv2d(x, weight_Phi, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups)
# # # if __name__ == "__main__":
# # #     # 创建一个输入张量，模拟一个 1 张 3 通道的 64x64 图像
# # #     x = torch.randn(1, 3, 64, 64)  # shape: (B, C, H, W)
# # #
# # #     # 设置参数
# # #     in_channels = 3
# # #     out_channels = 16
# # #     kernel_size = 5
# # #     den = [0.25, 0.5]  # len(den) = 2 → alfa长度=5 → kernel_size=5
# # #
# # #     # 创建 wConv2d 层
# # #     conv = wConv2d(
# # #         in_channels=in_channels,
# # #         out_channels=out_channels,
# # #         kernel_size=kernel_size,
# # #         den=den,
# # #         stride=1,
# # #         padding=2,
# # #         bias=False
# # #     )
# # #
# # #     # 前向传播
# # #     out = conv(x)
# # #
# # #     # 输出形状
# # #     print(f"Input shape:  {x.shape}")
# # #     print(f"Output shape: {out.shape}")
# #
# #
# import argparse
# import torch
# import torch.nn as nn
# from PIL import Image
# import torchvision.transforms as T
# import torchvision.utils as vutils
#
#
# #
# # class SimpleDenoise(nn.Module):
# #     def __init__(self, in_chans: int):
# #         super().__init__()
# #         # 1) 水下先验校正：初始化时给个经验值
# #         init_prior = torch.tensor([1.2, 1.0, 0.8])  # R,G,B 增益，可根据水下红光衰减严重程度调整
# #         self.prior = nn.Parameter(init_prior)
# #
# #         # 2) 残差去噪：可以复用你之前的 SimpleDenoise 或 wConv2d 模块
# #         self.denoise = SimpleDenoise(in_chans)
# #         self.net = nn.Sequential(
# #             # 第一层：线性平滑
# #             wConv2d(in_chans, in_chans, kernel_size = 3, den = [0.7]),
# #             nn.BatchNorm2d(in_chans),
# #             # 第二层：非线性变换
# #             wConv2d(in_chans, in_chans, kernel_size = 5, den = [0.2, 0.8],padding=2),
# #             nn.BatchNorm2d(in_chans),
# #             nn.ReLU(inplace=True),
# #
# #             # 第三层：再一次卷积恢复通道
# #             wConv2d(in_chans, in_chans, kernel_size = 3, den = [0.7]),
# #             nn.BatchNorm2d(in_chans),
# #
# #         )
# #
# #     def forward(self, x):
# #         # 残差连接：保留原始细节
# #         p = torch.clamp(self.prior, 0.5, 2.0)  # 限制在 [0.5,2.0] 避免过大
# #         x = x * p.view(1, 3, 1, 1)
# #
# #         # 再做残差去噪
# #         x = self.denoise(x)
# #         print(x.shape)
# #         return self.net(x)+x
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class wConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, den, stride=1, padding=None, groups=1, bias=False):
#         super().__init__()
#         self.stride = stride
#         self.padding = padding if padding is not None else kernel_size // 2
#         self.groups = groups
#
#         # 卷积权重和偏置
#         self.weight = nn.Parameter(torch.empty(
#             out_channels, in_channels // groups, kernel_size, kernel_size))
#         nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
#         self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
#
#         # 物理启发：通道模调因子（按输出通道）
#         self.channel_mod = nn.Parameter(torch.ones(out_channels))
#
#         # 构造 Phi 核
#         den = torch.as_tensor(den, dtype=torch.float32)
#         alfa = torch.cat([den, den.new_tensor([1.0]), den.flip(0)])
#         Phi = alfa[:, None] * alfa[None, :]
#         if Phi.shape != (kernel_size, kernel_size):
#             raise ValueError(f"Phi shape {Phi.shape} 必须匹配 kernel_size {kernel_size}")
#         self.register_buffer('Phi', Phi)
#
#     def forward(self, x):
#         # 逐输出通道做物理调制
#         Phi = self.Phi.to(x.device)
#         weight = self.weight * Phi[None, None, :, :]  # ↳ 按空间位置做 Φ 调制
#         weight = weight * self.channel_mod.view(-1, 1, 1, 1)  # ↳ 按通道做调制
#         return F.conv2d(x, weight, bias=self.bias,
#                         stride=self.stride, padding=self.padding,
#                         groups=self.groups)
#
#
# class UnderwaterDenoise(nn.Module):
#     def __init__(self, in_chans: int = 3):
#         super().__init__()
#         # 1) 水下色散先验：R/G/B 增益，可根据实际场景微调
#         init_prior = torch.tensor([1.2, 1.0, 0.8])
#         self.prior = nn.Parameter(init_prior)
#
#         # 2) 两层物理启发 wConv2d + 残差
#         self.layer1 = nn.Sequential(
#             wConv2d(in_chans, 64, kernel_size=5, den=[0.3, 0.6]),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#         )
#         self.layer2 = nn.Sequential(
#             wConv2d(64, in_chans, kernel_size=3, den=[0.7]),
#             nn.BatchNorm2d(in_chans),
#         )
#
#     def forward(self, x):
#         # （a）先验校正：对每个通道按物理增益做缩放
#         gain = torch.clamp(self.prior, 0.5, 2.0)
#         x_corr = x * gain.view(1, -1, 1, 1)
#
#         # （b）残差去噪
#         feat = self.layer1(x_corr)
#         rec  = self.layer2(feat)
#         return x_corr + rec
#
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Test SimpleDenoise on an image")
#     parser.add_argument("input_image", type=str, help="Path to the input image")
#     parser.add_argument("output_image", type=str, help="Path to save the denoised image")
#     args = parser.parse_args()
#
#     # 1. 读取并预处理
#     to_tensor = T.ToTensor()
#     to_pil = T.ToPILImage()
#     img = Image.open(args.input_image).convert("RGB")
#     x = to_tensor(img).unsqueeze(0)  # [1, 3, H, W]
#
#     # 2. 构建模型并推理
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = UnderwaterDenoise(in_chans=3).to(device)
#     model.eval()
#     with torch.no_grad():
#         x = x.to(device)
#         denoised = model(x)
#
#     # 3. 保存结果
#     denoised_img = denoised.squeeze(0).cpu()
#     vutils.save_image(denoised_img, args.output_image)
#
#     print(f"Denoised image saved to {args.output_image}")
#
#
