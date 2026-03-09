from mmengine.model import BaseModule
from mmdet.registry import MODELS
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .models.vmamba import Backbone_VSSM
from .models.vmamba import SS2D
from .models.vmamba import VSSBlock
from mmcv.cnn import build_norm_layer


@MODELS.register_module()
class MM_VSSM(BaseModule, Backbone_VSSM):
    def __init__(self, *args, **kwargs):
        BaseModule.__init__(self)
        Backbone_VSSM.__init__(self, *args, **kwargs)


@MODELS.register_module()
class ASS2D(SS2D):
    def __init__(self, *args, **kwargs):
        SS2D.__init__(self, *args, **kwargs)


@MODELS.register_module()
class VSS(VSSBlock):
    def __init__(self, *args, **kwargs):
        VSSBlock.__init__(self, *args, **kwargs)


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


class MBWTConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=5,
        stride=1,
        bias=True,
        wt_levels=1,
        wt_type='db1',
        ssm_ratio=1,
        forward_type="v05",
    ):
        super(MBWTConv2d, self).__init__()

        # 不再强制 in_channels == out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        # 构建小波变换和逆变换的滤波器，并固定（requires_grad=False）
        self.wt_filter, self.iwt_filter = create_wavelet_filter(
            wt_type, in_channels, in_channels, torch.float
        )
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        # 用于快速调用的小波函数
        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

        # 全局状态空间建模：输入通道是 in_channels，输出也是 in_channels
        self.global_atten = ASS2D(
            d_model=in_channels,
            d_state=1,
            ssm_ratio=ssm_ratio,
            initialize="v2",
            forward_type=forward_type,
            channel_first=True,
            k_group=2
        )
        # 基础尺度变换也是针对 in_channels
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])
        # 小波域 depthwise 卷积：每层都对 in_channels * 4 做 depthwise
        self.wavelet_convs = nn.ModuleList([
            nn.Conv2d(
                in_channels * 4,
                in_channels * 4,
                kernel_size,
                padding='same',
                stride=1,
                dilation=1,
                groups=in_channels * 4,
                bias=False
            )
            for _ in range(self.wt_levels)
        ])
        # 对应的小波尺度缩放（可学习）
        self.wavelet_scale = nn.ModuleList([
            _ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1)
            for _ in range(self.wt_levels)
        ])

        # 如果需要下采样，则在融合后通过 depthwise conv 做 stride
        if self.stride > 1:
            # 注意：这里要对 out_channels 做 depthwise
            # 因此 weight 尺寸是 [out_channels, 1, 1, 1]，groups=out_channels
            self.stride_filter = nn.Parameter(
                torch.ones(out_channels, 1, 1, 1),
                requires_grad=False
            )
            self.do_stride = lambda x_in: F.conv2d(
                x_in,
                self.stride_filter,
                bias=None,
                stride=self.stride,
                groups=out_channels
            )
        else:
            self.do_stride = None

        # 可学习门控：融合 x 与 x_tag，用 sigmoid 控制比例
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: [B, in_channels, H, W]
        返回: [B, out_channels, H//stride, W//stride]
        """
        # =========================
        # 1. 小波分解 + 处理
        # =========================
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x  # 初始 LL 分支是输入

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape  # [B, C, H, W]
            shapes_in_levels.append(curr_shape)

            # 如果 H 或 W 是奇数，通过 pad 使得能被 2 整除
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            # 执行小波正向变换：输出尺寸 [B, C, 4, H//2, W//2]
            curr_x = self.wt_function(curr_x_ll)
            # 提取 LL 分量作为下一层的小波输入
            curr_x_ll = curr_x[:, :, 0, :, :]

            # 把 4 个子带 reshape 到 “通道堆叠” 形式，方便做 depthwise 卷积
            shape_x = curr_x.shape  # [B, C, 4, H', W']
            # reshape → [B, C*4, H', W']
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])

            # depthwise 卷积 + 尺度缩放
            curr_x_tag = self.wavelet_scale[i](
                self.wavelet_convs[i](curr_x_tag)
            )
            # 再 reshape 回 [B, C, 4, H', W']
            curr_x_tag = curr_x_tag.reshape(shape_x)

            # 保存本层：LL 分量和高频分量（LH/HL/HH）
            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])    # [B, C, H', W']
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])   # [B, C, 3, H', W']

        # =========================
        # 2. 小波逆变换——自底向上重建
        # =========================
        next_x_ll = 0
        # 从最后一层往前逐层逆变换
        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()   # 上一层的 LL
            curr_x_h = x_h_in_levels.pop()     # 对应的高频
            curr_shape = shapes_in_levels.pop()

            # 跨层残差：把上一层的重建（next_x_ll）加到当前层 LL
            curr_x_ll = curr_x_ll + next_x_ll

            # 拼回 [LL, LH, HL, HH]
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            # 逆小波
            next_x_ll = self.iwt_function(curr_x)
            # 有可能尺寸被 pad，需要裁剪回原始 H×W
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        # next_x_ll: [B, in_channels, H, W] —— 这是小波分支输出
        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        # =========================
        # 3. 全局状态空间建模 & 基础尺度缩放
        # =========================
        # global_atten(x) 的输出尺寸是 [B, in_channels, H, W]
        x = self.base_scale(self.global_atten(x))  # 先全局建模，再尺度调节

        # 和小波分支做残差融合
        alpha = self.fusion_gate(x_tag)
        x = x * (1 - alpha) + x_tag * alpha

        # =========================
        # 5. 如果需要下采样，则在投影后对 out_channels 做 depthwise stride
        # =========================
        if self.do_stride is not None:
            x = self.do_stride(x)  # [B, out_channels, H//stride, W//stride]

        return x


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


if __name__ == '__main__':
    x = torch.rand(1, 96, 64, 64).cuda()
    m = MBWTConv2d(96, 96, 3, wt_levels=2).cuda()
    print(m)
    y = m(x).cuda()
    # y = d(x)
    print(y.shape)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = build_norm_layer(norm_cfg, b)[1]
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


def get_shape(tensor):
    shape = tensor.shape
    if torch.onnx.is_in_onnx_export():
        shape = [i.cpu().numpy() for i in shape]
    return shape


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv2d_BN(in_features, hidden_features, norm_cfg=norm_cfg)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = Conv2d_BN(hidden_features, out_features, norm_cfg=norm_cfg)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads,
                 attn_ratio=4,
                 activation=None,
                 norm_cfg=dict(type='BN', requires_grad=True), ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)

        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))

    def forward(self, x):  # x (B,N,C)
        B, C, H, W = get_shape(x)

        qq = self.to_q(x).reshape(B, self.num_heads, self.key_dim, H * W).permute(0, 1, 3, 2)
        kk = self.to_k(x).reshape(B, self.num_heads, self.key_dim, H * W)
        vv = self.to_v(x).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2)

        attn = torch.matmul(qq, kk)
        attn = attn.softmax(dim=-1)  # dim = k

        xx = torch.matmul(attn, vv)

        xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H, W)
        xx = self.proj(xx)
        return xx


class top_Block(nn.Module):

    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_cfg=dict(type='BN2d', requires_grad=True)):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.attn = Attention(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio, activation=act_layer,
                              norm_cfg=norm_cfg)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                       norm_cfg=norm_cfg)

    def forward(self, x1):
        x1 = x1 + self.drop_path(self.attn(x1))
        x1 = x1 + self.drop_path(self.mlp(x1))
        return x1


class Mamba(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.,
                 drop_path=0.1, act_layer=nn.ReLU, norm_cfg=dict(type='BN2d', requires_grad=True)):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.attn = ASS2D(
            d_model=dim,
            d_state=1,
            ssm_ratio=1.0,
            initialize="v2",
            forward_type="v05",
            channel_first=True,
            k_group=2
        )

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                       norm_cfg=norm_cfg)

    def forward(self, x1):
        x1 = x1 + self.drop_path(self.attn(x1))
        x1 = x1 + self.drop_path(self.mlp(x1))
        return x1


class TopBasicLayer(nn.Module):
    def __init__(self, block_num, embedding_dim, key_dim, num_heads,
                 mlp_ratio=4., attn_ratio=2., drop=0., attn_drop=0., drop_path=0.,
                 norm_cfg=dict(type='BN2d', requires_grad=True),
                 act_layer=nn.ReLU6):
        super().__init__()
        self.block_num = block_num

        self.transformer_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.transformer_blocks.append(top_Block(
                embedding_dim, key_dim=key_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, attn_ratio=attn_ratio,
                drop=drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_cfg=norm_cfg, act_layer=act_layer))

    def forward(self, x):
        # token * N
        for i in range(self.block_num):
            x = self.transformer_blocks[i](x)
        return x



