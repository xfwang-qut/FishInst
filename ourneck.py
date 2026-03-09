import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, MultiConfig
from torch import Tensor
from .FreqFusion import FreqFusion
from .custom import MBWTConv2d

@MODELS.register_module()
class UWFPN(BaseModule):
    def __init__(
        self,
        in_channels: list[int],
        out_channels: int,
        num_outs: int,
        eps: float = 1e-4,
        start_level: int = 0,
        end_level: int = -1,
        add_extra_convs: bool | str = False,
        relu_before_extra_convs: bool = False,
        no_norm_on_lateral: bool = False,
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        act_cfg: OptConfigType = None,
        upsample_cfg: ConfigType = dict(mode='nearest'),
        init_cfg: MultiConfig = dict(
            type='Xavier', layer='Conv2d', distribution='uniform')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.start_level = start_level
        self.end_level = end_level
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.eps = eps
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1

        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:
            add_extra_convs = 'on_input'
        self.add_extra_convs = add_extra_convs
        # lateral and fpn convs
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            self.lateral_convs.append(
                ConvModule(
                    in_channels[i], out_channels, 1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg if not no_norm_on_lateral else None,
                    act_cfg=act_cfg,
                    inplace=False
                )
            )
            self.fpn_convs.append(
                nn.Sequential(
                    MBWTConv2d(out_channels, out_channels, 3,wt_levels=2),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                )
            )

        # extra convolutional levels
        extra_levels = num_outs - (self.backbone_end_level - start_level)
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                in_ch = (
                    in_channels[self.backbone_end_level - 1]
                    if i == 0 and self.add_extra_convs == 'on_input'
                    else out_channels
                )
                self.fpn_convs.append(
                    ConvModule(
                        in_ch, out_channels, 3, stride=2, padding=1,
                        conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                        act_cfg=act_cfg, inplace=False
                    )
                )

        # frequency fusion modules per adjacent level
        self.freq_fusions = nn.ModuleList()
        levels = self.backbone_end_level - self.start_level
        for _ in range(levels - 1):
            self.freq_fusions.append(
                FreqFusion(
                    hr_channels=out_channels,
                    lr_channels=out_channels,
                    scale_factor=1,
                    lowpass_kernel=5,
                    highpass_kernel=3,
                    up_group=1,
                    upsample_mode='nearest',
                    align_corners=False,
                    feature_resample=False,
                    feature_resample_group=8,
                    hr_residual=True,
                    comp_feat_upsample=True,
                    compressed_channels=(out_channels * 2) // 8,
                    use_high_pass=True,
                    use_low_pass=True,
                    semi_conv=True,
                    feature_resample_norm=True
                )
            )

        # learnable weights for fusion
        self.w1 = nn.Parameter(torch.ones(out_channels))
        self.w2 = nn.Parameter(torch.ones(out_channels))

    def forward(self, inputs: tuple[Tensor]) -> tuple[Tensor, ...]:
        assert len(inputs) == len(self.in_channels)

        # build lateral features
        laterals = [
            lat_conv(inputs[i + self.start_level])
            for i, lat_conv in enumerate(self.lateral_convs)
        ]

        # top-down path with frequency fusion and weighted merge
        for idx in range(len(laterals) - 1, 0, -1):
            fusion = self.freq_fusions[idx - 1]
            _, lr_fused, hr_fused = fusion(
                hr_feat=laterals[idx - 1], lr_feat=laterals[idx]
            )
            # weighted fusion between low-frequency and high-frequency outputs
            w1 = F.relu(self.w1).view(1, -1, 1, 1)
            w2 = F.relu(self.w2).view(1, -1, 1, 1)
            fused = (w1 * lr_fused + w2 * hr_fused) / (w1 + w2 + self.eps)
            laterals[idx - 1] = fused

        # build outputs
        outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]

        # handle extra outputs beyond backbone levels
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for _ in range(self.num_outs - len(outs)):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                if self.add_extra_convs == 'on_input':
                    source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[len(laterals)](source))
                for i in range(len(laterals) + 1, self.num_outs):
                    feat = F.relu(outs[-1]) if self.relu_before_extra_convs else outs[-1]
                    outs.append(self.fpn_convs[i](feat))

        return tuple(outs)
