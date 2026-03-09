# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from copy import deepcopy
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmengine.logging import MMLogger
from mmengine.model.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init)
from mmengine.runner.checkpoint import CheckpointLoader
import warnings
import torch.utils.checkpoint as cp
from mmdet.models.layers import PatchEmbed
from mmengine.model import ModuleList
from mmengine.utils import to_2tuple
from mmdet.registry import MODELS
from mmengine.model import BaseModule
from mmcv.cnn import build_norm_layer
import torch.nn as nn
import torch
import torch.nn.functional as F
from mmdet.models.layers import PatchMerging


class wConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, den, stride=1, padding=None, groups=1, bias=False):
        super().__init__()
        self.stride = stride
        self.padding = padding if padding is not None else kernel_size // 2
        self.groups = groups

        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels // groups, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        self.channel_mod = nn.Parameter(torch.ones(out_channels))
        den = torch.as_tensor(den, dtype=torch.float32)
        alfa = torch.cat([den, den.new_tensor([1.0]), den.flip(0)])
        Phi = alfa[:, None] * alfa[None, :]
        if Phi.shape != (kernel_size, kernel_size):
            raise ValueError(f"Phi shape {Phi.shape} 必须匹配 kernel_size {kernel_size}")
        self.register_buffer('Phi', Phi)

    def forward(self, x):
        Phi = self.Phi.to(x.device)
        weight = self.weight * Phi[None, None, :, :]
        weight = weight * self.channel_mod.view(-1, 1, 1, 1)
        return F.conv2d(x, weight, bias=self.bias,
                        stride=self.stride, padding=self.padding,
                        groups=self.groups)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, den):
        super().__init__()
        self.conv1 = wConv2d(in_ch, out_ch, kernel_size=3, den=den, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.act1  = nn.LeakyReLU(0.1, True)

        self.conv2 = wConv2d(out_ch, out_ch, kernel_size=3, den=den, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.act2  = nn.LeakyReLU(0.1, True)

        # 下采样
        self.pool = nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        skip = x
        x = self.pool(x)
        return x, skip


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, den):
        super().__init__()
        self.up    = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # 注意这里 in_ch 一定要等于 cat(x, skip) 后的通道数
        self.conv1 = wConv2d(in_ch,  out_ch, kernel_size=3, den=den, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.act1  = nn.LeakyReLU(0.1, True)

        self.conv2 = wConv2d(out_ch, out_ch, kernel_size=3, den=den, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.act2  = nn.LeakyReLU(0.1, True)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        return x


class UnderwaterDenoiseUNet(nn.Module):
    def __init__(self, in_chans=3, base_ch=32, depth=4, den_list=None):
        super().__init__()
        if den_list is None:
            den_list = [[0.5]] * depth

        # 1) Encoder
        self.downs = nn.ModuleList()
        chs = [in_chans] + [base_ch * (2**i) for i in range(depth)]
        for i in range(depth):
            self.downs.append(Down(chs[i], chs[i+1], den_list[i]))

        # 2) Bottleneck
        self.bot_conv1 = wConv2d(chs[-1], chs[-1]*2, kernel_size=3,
                                den=den_list[-1], padding=1)
        self.bot_bn1   = nn.BatchNorm2d(chs[-1]*2)
        self.bot_act1  = nn.LeakyReLU(0.1, True)
        self.bot_conv2 = wConv2d(chs[-1]*2, chs[-1], kernel_size=3,
                                den=den_list[-1], padding=1)
        self.bot_bn2   = nn.BatchNorm2d(chs[-1])
        self.bot_act2  = nn.LeakyReLU(0.1, True)

        # 3) Decoder：动态计算每层的 in_ch = 当前 x_ch + 对应 skip_ch
        self.ups = nn.ModuleList()
        skip_chs = chs[1:]            # [32,64,128,256]
        dec_chs  = skip_chs[::-1]     # [256,128,64,32]
        x_ch     = chs[-1]            # start from bottleneck 输出通道=256
        for i, skip_c in enumerate(dec_chs):
            out_c = skip_c
            in_c  = x_ch + skip_c
            self.ups.append(Up(in_c, out_c, den_list[-1-i]))
            x_ch = out_c             # 下一层的 x_ch 就是上一次的 out_c

        # 4) 最后把 x_ch (== base_ch) 卷回输入通道，并做残差融合
        self.final_conv = nn.Conv2d(base_ch, in_chans, kernel_size=1)
        self.alpha      = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        inp = x  # <— 保留输入图像
        skips = []
        for down in self.downs:
            x, skip = down(x)
            skips.append(skip)

        # Bottleneck
        x = self.bot_act1(self.bot_bn1(self.bot_conv1(x)))
        x = self.bot_act2(self.bot_bn2(self.bot_conv2(x)))

        # Decoder
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)

        rec = self.final_conv(x)  # → (B, in_chans, H, W)
        s = torch.sigmoid(self.alpha)
        # 用最开始的输入 inp 做残差融合，而不是 x
        return (1 - s) * inp + s * rec



import torch
from PIL import Image
import torchvision.transforms as T
import os

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt



# =========================== 用户可修改区域 ===========================

# 输入图像路径（改成你的测试图像）
INPUT_IMAGE = "D:/project/ma/condinstmam/Fish/train/007081.jpg"

# 去噪后图像保存路径
OUTPUT_IMAGE = "D:/project/ma/condinstmam/results/denoised.png"

# ResNet50 layer2 特征图可视化保存路径（彩色热力图）
HEATMAP_ORIG_PATH = "D:/project/ma/condinstmam/results/resnet_layer2_orig.png"
HEATMAP_DENOISED_PATH = "D:/project/ma/condinstmam/results/resnet_layer2_denoised.png"

# 训练好的 UNet 权重
CKPT_PATH = "D:/project/ma/condinstmam/weights/underwater_unet.pth"

# 推理设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ====================================================================


def load_unet():
    """构建模型并可选加载权重"""
    model = UnderwaterDenoiseUNet(in_chans=3, base_ch=32, depth=4).to(DEVICE)

    if CKPT_PATH and os.path.isfile(CKPT_PATH):
        print(f"[Info] 加载 UNet 权重: {CKPT_PATH}")
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
        # 兼容 {'state_dict': ...} 的情况
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        model.load_state_dict(ckpt, strict=True)
        print("[Info] UNet 权重加载完成")
    else:
        print("[Info] 未加载权重，使用随机初始化模型（用于测试网络结构是否能跑通）")

    model.eval()
    return model


def load_resnet_layer2():
    """构建 ResNet50，并截取到 layer2 作为特征提取网络"""
    try:
        resnet = models.resnet50(pretrained=True)
    except TypeError:
        # 兼容新版 torchvision 的写法
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet.to(DEVICE)
    resnet.eval()

    # conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2
    layer0_net = nn.Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool
    )
    layer2_net = nn.Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        resnet.layer1,
        resnet.layer2,
    )

    return layer2_net


def load_image(path):
    """读取并转换成 tensor"""
    img = Image.open(path).convert("RGB")
    transform = T.ToTensor()
    tensor = transform(img).unsqueeze(0).to(DEVICE)  # (1,3,H,W)
    return img, tensor


def tensor_to_image(tensor):
    """tensor → PIL Image"""
    if tensor.dim() == 4:
        tensor = tensor[0]
    tensor = torch.clamp(tensor, 0, 1)
    to_pil = T.ToPILImage()
    return to_pil(tensor.cpu())


def extract_resnet_layer2_feat(layer2_net, x):
    """
    x: (B,3,H,W)，值范围 [0,1]
    返回：layer2 输出特征 (B,C,H',W')
    """
    # ResNet50 期望输入 224x224，这里插值过去
    x_resized = F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False)

    # ImageNet 均值方差标准化
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    x_norm = (x_resized - mean) / std

    with torch.no_grad():
        feat = layer2_net(x_norm)  # (B,C,H',W')

    return feat


def save_heatmap(feat, save_path):
    """
    将 ResNet 特征绘制成彩色热力图并保存
    feat: (B,C,H,W)
    """
    # 这里用通道平均，也可以改成 feat[0, 某个通道]
    fmap = feat[0].mean(dim=0)  # (H,W)

    # min-max 归一化到 [0,1]
    fmap = fmap - fmap.min()
    if fmap.max() > 0:
        fmap = fmap / fmap.max()

    fmap_np = fmap.cpu().numpy()

    plt.figure(figsize=(3, 3))
    plt.imshow(fmap_np, cmap='jet')  # jet 伪彩色，类似你给的效果
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"[Info] 热力图已保存至: {save_path}")


def main():
    # 创建输出目录
    os.makedirs(os.path.dirname(OUTPUT_IMAGE) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(HEATMAP_ORIG_PATH) or ".", exist_ok=True)

    # 1) 加载 UNet 和 ResNet50-layer2
    unet = load_unet()
    layer2_net = load_resnet_layer2()

    # 2) 加载原始图像
    print(f"[Info] 读取图像: {INPUT_IMAGE}")
    _, inp = load_image(INPUT_IMAGE)

    # 3) 前向推理去噪
    print("[Info] 正在去噪...")
    with torch.no_grad():
        denoised = unet(inp)  # (1,3,H,W)

    # 4) 保存去噪图像
    out_img = tensor_to_image(denoised)
    out_img.save(OUTPUT_IMAGE)
    print(f"[Info] 去噪完成，图像已保存至: {OUTPUT_IMAGE}")

    # 5) 计算 ResNet50 第二层特征（原图 & 去噪图）
    print("[Info] 计算 ResNet50 layer2 特征...")
    feat_orig = extract_resnet_layer2_feat(layer2_net, inp)
    feat_denoised = extract_resnet_layer2_feat(layer2_net, denoised)

    print(f"[Info] 原图特征形状: {feat_orig.shape}")
    print(f"[Info] 去噪图特征形状: {feat_denoised.shape}")

    # 6) 保存彩色热力图
    save_heatmap(feat_orig, HEATMAP_ORIG_PATH)
    save_heatmap(feat_denoised, HEATMAP_DENOISED_PATH)


if __name__ == "__main__":
    main()




class WindowMSA(BaseModule):

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 init_cfg=None):

        super().__init__()
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5
        self.init_cfg = init_cfg

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # About 2x faster than original impl
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        Args:

            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


class ShiftWindowMSA(BaseModule):

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0,
                 proj_drop_rate=0,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 init_cfg=None):
        super().__init__(init_cfg)

        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            init_cfg=None)

        self.drop = build_dropout(dropout_layer)

    def forward(self, query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        # cyclic shift
        if self.shift_size > 0:
            shifted_query = torch.roll(
                query,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2))

            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(
                                                  attn_mask == 0, float(0.0))
        else:
            shifted_query = query
            attn_mask = None

        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(shifted_query)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, self.window_size**2, C)

        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)

        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = self.drop(x)
        return x

    def window_reverse(self, windows, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows


class SwinBlock(BaseModule):

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 window_size=7,
                 shift=False,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):

        super(SwinBlock, self).__init__()

        self.init_cfg = init_cfg
        self.with_cp = with_cp

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = ShiftWindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            init_cfg=None)

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=2,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=True,
            init_cfg=None)

    def forward(self, x, hw_shape):

        def _inner_forward(x):
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape)

            x = x + identity

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


class SwinBlockSequence(BaseModule):

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 depth,
                 window_size=7,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 downsample=None,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]

        self.blocks = ModuleList()
        for i in range(depth):
            block = SwinBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                window_size=window_size,
                shift=False if i % 2 == 0 else True,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None)
            self.blocks.append(block)

        self.downsample = downsample

    def forward(self, x, hw_shape):
        for block in self.blocks:
            x = block(x, hw_shape)

        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape


@MODELS.register_module()
class ASwinTransformer(BaseModule):

    def __init__(self,
                 pretrain_img_size=224,
                 in_channels=3,
                 embed_dims=96,
                 patch_size=4,
                 window_size=7,
                 mlp_ratio=4,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 strides=(4, 2, 2, 2),
                 out_indices=(0, 1, 2, 3),
                 qkv_bias=True,
                 qk_scale=None,
                 patch_norm=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 use_abs_pos_embed=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 pretrained=None,
                 convert_weights=False,
                 frozen_stages=-1,
                 init_cfg=None):
        self.convert_weights = convert_weights
        self.frozen_stages = frozen_stages
        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            assert len(pretrain_img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(pretrain_img_size)}'

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')

        super(ASwinTransformer, self).__init__(init_cfg=init_cfg)

        num_layers = len(depths)
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed

        assert strides[0] == patch_size, 'Use non-overlapping patch embed.'

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=strides[0],
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None)

        if self.use_abs_pos_embed:
            patch_row = pretrain_img_size[0] // patch_size
            patch_col = pretrain_img_size[1] // patch_size
            num_patches = patch_row * patch_col
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros((1, num_patches, embed_dims)))

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # set stochastic depth decay rule
        total_depth = sum(depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]

        self.stages = ModuleList()
        in_channels = embed_dims
        for i in range(num_layers):
            if i < num_layers - 1:
                downsample = PatchMerging(
                    in_channels=in_channels,
                    out_channels=2 * in_channels,
                    stride=strides[i + 1],
                    norm_cfg=norm_cfg if patch_norm else None,
                    init_cfg=None)
            else:
                downsample = None

            stage = SwinBlockSequence(
                embed_dims=in_channels,
                num_heads=num_heads[i],
                feedforward_channels=mlp_ratio * in_channels,
                depth=depths[i],
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=downsample,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None)
            self.stages.append(stage)
            if downsample:
                in_channels = downsample.out_channels

        self.num_features = [int(embed_dims * 2**i) for i in range(num_layers)]
        # Add a norm layer for each output
        for i in out_indices:
            layer = build_norm_layer(norm_cfg, self.num_features[i])[1]
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

        self.un = UnderwaterDenoiseUNet(in_chans=3, base_ch=32, depth=4)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(ASwinTransformer, self).train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            if self.use_abs_pos_embed:
                self.absolute_pos_embed.requires_grad = False
            self.drop_after_pos.eval()

        for i in range(1, self.frozen_stages + 1):

            if (i - 1) in self.out_indices:
                norm_layer = getattr(self, f'norm{i-1}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

            m = self.stages[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        logger = MMLogger.get_current_instance()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            if self.use_abs_pos_embed:
                trunc_normal_(self.absolute_pos_embed, std=0.02)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt
            if self.convert_weights:
                # supported loading weight from original repo,
                _state_dict = swin_converter(_state_dict)

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # reshape absolute position embedding
            if state_dict.get('absolute_pos_embed') is not None:
                absolute_pos_embed = state_dict['absolute_pos_embed']
                N1, L, C1 = absolute_pos_embed.size()
                N2, C2, H, W = self.absolute_pos_embed.size()
                if N1 != N2 or C1 != C2 or L != H * W:
                    logger.warning('Error in loading absolute_pos_embed, pass')
                else:
                    state_dict['absolute_pos_embed'] = absolute_pos_embed.view(
                        N2, H, W, C2).permute(0, 3, 1, 2).contiguous()

            # interpolate position bias table if needed
            relative_position_bias_table_keys = [
                k for k in state_dict.keys()
                if 'relative_position_bias_table' in k
            ]
            for table_key in relative_position_bias_table_keys:
                table_pretrained = state_dict[table_key]
                table_current = self.state_dict()[table_key]
                L1, nH1 = table_pretrained.size()
                L2, nH2 = table_current.size()
                if nH1 != nH2:
                    logger.warning(f'Error in loading {table_key}, pass')
                elif L1 != L2:
                    S1 = int(L1**0.5)
                    S2 = int(L2**0.5)
                    table_pretrained_resized = F.interpolate(
                        table_pretrained.permute(1, 0).reshape(1, nH1, S1, S1),
                        size=(S2, S2),
                        mode='bicubic')
                    state_dict[table_key] = table_pretrained_resized.view(
                        nH2, L2).permute(1, 0).contiguous()

            # load state_dict
            self.load_state_dict(state_dict, False)

    def forward(self, x):
        x = self.un(x)
        x, hw_shape = self.patch_embed(x)

        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,
                               self.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(out)

        return outs


def swin_converter(ckpt):

    new_ckpt = OrderedDict()

    def correct_unfold_reduction_order(x):
        out_channel, in_channel = x.shape
        x = x.reshape(out_channel, 4, in_channel // 4)
        x = x[:, [0, 2, 1, 3], :].transpose(1,
                                            2).reshape(out_channel, in_channel)
        return x

    def correct_unfold_norm_order(x):
        in_channel = x.shape[0]
        x = x.reshape(4, in_channel // 4)
        x = x[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)
        return x

    for k, v in ckpt.items():
        if k.startswith('head'):
            continue
        elif k.startswith('layers'):
            new_v = v
            if 'attn.' in k:
                new_k = k.replace('attn.', 'attn.w_msa.')
            elif 'mlp.' in k:
                if 'mlp.fc1.' in k:
                    new_k = k.replace('mlp.fc1.', 'ffn.layers.0.0.')
                elif 'mlp.fc2.' in k:
                    new_k = k.replace('mlp.fc2.', 'ffn.layers.1.')
                else:
                    new_k = k.replace('mlp.', 'ffn.')
            elif 'downsample' in k:
                new_k = k
                if 'reduction.' in k:
                    new_v = correct_unfold_reduction_order(v)
                elif 'norm.' in k:
                    new_v = correct_unfold_norm_order(v)
            else:
                new_k = k
            new_k = new_k.replace('layers', 'stages', 1)
        elif k.startswith('patch_embed'):
            new_v = v
            if 'proj' in k:
                new_k = k.replace('proj', 'projection')
            else:
                new_k = k
        else:
            new_v = v
            new_k = k

        new_ckpt['backbone.' + new_k] = new_v

    return new_ckpt

