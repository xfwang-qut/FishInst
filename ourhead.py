# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional, Tuple
from mmdet.models.layers import NormedConv2d
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale
from mmengine.config import ConfigDict
from mmengine.model import BaseModule, kaiming_init
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures.bbox import cat_boxes
from mmdet.utils import (ConfigType, InstanceList, MultiConfig, OptConfigType,
                         OptInstanceList, reduce_mean)
from mmdet.models.task_modules.prior_generators import MlvlPointGenerator
from mmdet.models.utils import (aligned_bilinear, filter_scores_and_topk, multi_apply,
                                relative_coordinate_maps, select_single_mlvl, empty_instances)
from mmdet.models.dense_heads.base_mask_head import BaseMaskHead
from mmdet.models.dense_heads.fcos_head import FCOSHead
from .custom import TopBasicLayer, ASS2D, Mamba

INF = 1e8
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        out = avg_out + max_out
        return torch.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return torch.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

@MODELS.register_module()
class UWCondInstBboxHead(FCOSHead):

    def __init__(self, *args, num_params: int = 169, **kwargs) -> None:
        self.num_params = num_params
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        super()._init_layers()
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.controller = nn.Conv2d(
            self.feat_channels, self.num_params, 3, padding=1)

    def forward_single(self, x: Tensor, scale: Scale,
                       stride: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        cls_score, bbox_pred, cls_feat, reg_feat = \
            super(FCOSHead, self).forward_single(x)
        if self.centerness_on_reg:
            centerness = self.conv_centerness(cls_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            # bbox_pred needed for gradient computation has been modified
            # by F.relu(bbox_pred) when run with PyTorch 1.10. So replace
            # F.relu(bbox_pred) with bbox_pred.clamp(min=0)
            bbox_pred = bbox_pred.clamp(min=0)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        param_pred = self.controller(cls_feat)
        return cls_score, bbox_pred, centerness, param_pred

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            centernesses: List[Tensor],
            param_preds: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        # Need stride for rel coord compute
        all_level_points_strides = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device,
            with_stride=True)
        all_level_points = [i[:, :2] for i in all_level_points_strides]
        all_level_strides = [i[:, 2] for i in all_level_points_strides]
        labels, bbox_targets, pos_inds_list, pos_gt_inds_list = \
            self.get_targets(all_level_points, batch_gt_instances)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_preds)
            pos_decoded_target_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        self._raw_positive_infos.update(cls_scores=cls_scores)
        self._raw_positive_infos.update(centernesses=centernesses)
        self._raw_positive_infos.update(param_preds=param_preds)
        self._raw_positive_infos.update(all_level_points=all_level_points)
        self._raw_positive_infos.update(all_level_strides=all_level_strides)
        self._raw_positive_infos.update(pos_gt_inds_list=pos_gt_inds_list)
        self._raw_positive_infos.update(pos_inds_list=pos_inds_list)

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness)

    def get_targets(
            self, points: List[Tensor], batch_gt_instances: InstanceList
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list, pos_inds_list, pos_gt_inds_list = \
            multi_apply(
                self._get_targets_single,
                batch_gt_instances,
                points=concat_points,
                regress_ranges=concat_regress_ranges,
                num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
        return (concat_lvl_labels, concat_lvl_bbox_targets, pos_inds_list,
                pos_gt_inds_list)

    def _get_targets_single(
            self, gt_instances: InstanceData, points: Tensor,
            regress_ranges: Tensor, num_points_per_lvl: List[int]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = len(gt_instances)
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        gt_masks = gt_instances.get('masks', None)

        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                gt_bboxes.new_zeros((num_points, 4)), \
                gt_bboxes.new_zeros((0,), dtype=torch.int64), \
                gt_bboxes.new_zeros((0,), dtype=torch.int64)

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
                gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            # if gt_mask not None, use gt mask's centroid to determine
            # the center region rather than gt_bbox center
            if gt_masks is None:
                center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
                center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            else:
                h, w = gt_masks.height, gt_masks.width
                masks = gt_masks.to_tensor(
                    dtype=torch.bool, device=gt_bboxes.device)
                yys = torch.arange(
                    0, h, dtype=torch.float32, device=masks.device)
                xxs = torch.arange(
                    0, w, dtype=torch.float32, device=masks.device)
                # m00/m10/m01 represent the moments of a contour
                # centroid is computed by m00/m10 and m00/m01
                m00 = masks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
                m10 = (masks * xxs).sum(dim=-1).sum(dim=-1)
                m01 = (masks * yys[:, None]).sum(dim=-1).sum(dim=-1)
                center_xs = m10 / m00
                center_ys = m01 / m00

                center_xs = center_xs[None].expand(num_points, num_gts)
                center_ys = center_ys[None].expand(num_points, num_gts)
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
                (max_regress_distance >= regress_ranges[..., 0])
                & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        # return pos_inds & pos_gt_inds
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().reshape(-1)
        pos_gt_inds = min_area_inds[labels < self.num_classes]
        return labels, bbox_targets, pos_inds, pos_gt_inds

    def get_positive_infos(self) -> InstanceList:
        assert len(self._raw_positive_infos) > 0

        pos_gt_inds_list = self._raw_positive_infos['pos_gt_inds_list']
        pos_inds_list = self._raw_positive_infos['pos_inds_list']
        num_imgs = len(pos_gt_inds_list)

        cls_score_list = []
        centerness_list = []
        param_pred_list = []
        point_list = []
        stride_list = []
        for cls_score_per_lvl, centerness_per_lvl, param_pred_per_lvl, \
                point_per_lvl, stride_per_lvl in \
                zip(self._raw_positive_infos['cls_scores'],
                    self._raw_positive_infos['centernesses'],
                    self._raw_positive_infos['param_preds'],
                    self._raw_positive_infos['all_level_points'],
                    self._raw_positive_infos['all_level_strides']):
            cls_score_per_lvl = \
                cls_score_per_lvl.permute(
                    0, 2, 3, 1).reshape(num_imgs, -1, self.num_classes)
            centerness_per_lvl = \
                centerness_per_lvl.permute(
                    0, 2, 3, 1).reshape(num_imgs, -1, 1)
            param_pred_per_lvl = \
                param_pred_per_lvl.permute(
                    0, 2, 3, 1).reshape(num_imgs, -1, self.num_params)
            point_per_lvl = point_per_lvl.unsqueeze(0).repeat(num_imgs, 1, 1)
            stride_per_lvl = stride_per_lvl.unsqueeze(0).repeat(num_imgs, 1)

            cls_score_list.append(cls_score_per_lvl)
            centerness_list.append(centerness_per_lvl)
            param_pred_list.append(param_pred_per_lvl)
            point_list.append(point_per_lvl)
            stride_list.append(stride_per_lvl)
        cls_scores = torch.cat(cls_score_list, dim=1)
        centernesses = torch.cat(centerness_list, dim=1)
        param_preds = torch.cat(param_pred_list, dim=1)
        all_points = torch.cat(point_list, dim=1)
        all_strides = torch.cat(stride_list, dim=1)

        positive_infos = []
        for i, (pos_gt_inds,
                pos_inds) in enumerate(zip(pos_gt_inds_list, pos_inds_list)):
            pos_info = InstanceData()
            pos_info.points = all_points[i][pos_inds]
            pos_info.strides = all_strides[i][pos_inds]
            pos_info.scores = cls_scores[i][pos_inds]
            pos_info.centernesses = centernesses[i][pos_inds]
            pos_info.param_preds = param_preds[i][pos_inds]
            pos_info.pos_assigned_gt_inds = pos_gt_inds
            pos_info.pos_inds = pos_inds
            positive_infos.append(pos_info)
        return positive_infos

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        score_factors: Optional[List[Tensor]] = None,
                        param_preds: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        with_nms: bool = True) -> InstanceList:
        assert len(cls_scores) == len(bbox_preds)

        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        all_level_points_strides = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device,
            with_stride=True)
        all_level_points = [i[:, :2] for i in all_level_points_strides]
        all_level_strides = [i[:, 2] for i in all_level_points_strides]

        result_list = []

        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            cls_score_list = select_single_mlvl(
                cls_scores, img_id, detach=True)
            bbox_pred_list = select_single_mlvl(
                bbox_preds, img_id, detach=True)
            if with_score_factors:
                score_factor_list = select_single_mlvl(
                    score_factors, img_id, detach=True)
            else:
                score_factor_list = [None for _ in range(num_levels)]
            param_pred_list = select_single_mlvl(
                param_preds, img_id, detach=True)

            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                score_factor_list=score_factor_list,
                param_pred_list=param_pred_list,
                mlvl_points=all_level_points,
                mlvl_strides=all_level_strides,
                img_meta=img_meta,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                param_pred_list: List[Tensor],
                                mlvl_points: List[Tensor],
                                mlvl_strides: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        if score_factor_list[0] is None:
            # e.g. Retina, FreeAnchor, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, etc.
            with_score_factors = True

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bbox_preds = []
        mlvl_param_preds = []
        mlvl_valid_points = []
        mlvl_valid_strides = []
        mlvl_scores = []
        mlvl_labels = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None
        for level_idx, (cls_score, bbox_pred, score_factor,
                        param_pred, points, strides) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                              score_factor_list, param_pred_list,
                              mlvl_points, mlvl_strides)):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            dim = self.bbox_coder.encode_size
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, dim)
            if with_score_factors:
                score_factor = score_factor.permute(1, 2,
                                                    0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = cls_score.softmax(-1)[:, :-1]

            param_pred = param_pred.permute(1, 2,
                                            0).reshape(-1, self.num_params)

            score_thr = cfg.get('score_thr', 0)

            results = filter_scores_and_topk(
                scores, score_thr, nms_pre,
                dict(
                    bbox_pred=bbox_pred,
                    param_pred=param_pred,
                    points=points,
                    strides=strides))
            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            param_pred = filtered_results['param_pred']
            points = filtered_results['points']
            strides = filtered_results['strides']

            if with_score_factors:
                score_factor = score_factor[keep_idxs]

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_param_preds.append(param_pred)
            mlvl_valid_points.append(points)
            mlvl_valid_strides.append(strides)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = cat_boxes(mlvl_valid_points)
        bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)

        results = InstanceData()
        results.bboxes = bboxes
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)
        results.param_preds = torch.cat(mlvl_param_preds)
        results.points = torch.cat(mlvl_valid_points)
        results.strides = torch.cat(mlvl_valid_strides)
        if with_score_factors:
            results.score_factors = torch.cat(mlvl_score_factors)

        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)


class UWMaskFeatModule(BaseModule):

    def __init__(self,
                 in_channels: int,
                 feat_channels: int,
                 start_level: int,
                 end_level: int,
                 out_channels: int,
                 mask_stride: int = 4,
                 eps: float = 1e-4,
                 num_stacked_convs: int = 4,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,
                 init_cfg: MultiConfig = [
                     dict(type='Normal', layer='Conv2d', std=0.01)
                 ],
                 **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.start_level = start_level
        self.end_level = end_level
        self.mask_stride = mask_stride
        self.num_stacked_convs = num_stacked_convs
        assert start_level >= 0 and end_level >= start_level
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()
        self.eps = eps       

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.convs_all_levels = nn.ModuleList()
        for i in range(self.start_level, self.end_level + 1):
            convs_per_level = nn.Sequential()
            convs_per_level.add_module(
                f'conv{i}',
                ConvModule(
                    self.in_channels,
                    self.feat_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=False,
                    bias=False))
            self.convs_all_levels.append(convs_per_level)

        conv_branch = []
        for _ in range(self.num_stacked_convs):
            # conv_branch.append(
            #     ConvModule(
            #         self.feat_channels,
            #         self.feat_channels,
            #         3,
            #         padding=1,
            #         conv_cfg=self.conv_cfg,
            #         norm_cfg=self.norm_cfg,
            #         bias=False),
            #         CBAM(self.feat_channels)
            # )
            conv_branch.append(
                nn.Sequential(
                    ConvModule(
                        self.feat_channels,
                        self.feat_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        bias=False
                    ),
                    CBAM(self.feat_channels)
                )
            )            
        self.conv_branch = nn.Sequential(*conv_branch)

        self.conv_pred = nn.Conv2d(
            self.feat_channels, self.out_channels, 1, stride=1)

        self.w1conv = nn.Sequential(
            nn.Conv2d(self.feat_channels, self.feat_channels, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True)

        )

        self.w2conv = nn.Sequential(
            nn.Conv2d(self.feat_channels, self.feat_channels, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True)
        )
        
        self.w3conv = nn.Sequential(
            nn.Conv2d(self.feat_channels, self.feat_channels, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True)
        )
        dpr = [x.item() for x in torch.linspace(0, 0.1, 3)]
        self.atten = TopBasicLayer(
            block_num=3,
            embedding_dim=3*self.feat_channels,
            key_dim=8,
            num_heads=8,
            mlp_ratio=1,
            attn_ratio=2,
            drop=0, attn_drop=0,
            drop_path=dpr,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )
        self.attenconv = nn.Sequential(
            nn.Conv2d(3*self.feat_channels, self.feat_channels, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True) 
        )
        self.pool = nn.functional.adaptive_avg_pool2d

        self.wei1 = nn.Parameter(torch.tensor(1.0))
        self.wei2 = nn.Parameter(torch.tensor(1.0))
    
    def init_weights(self) -> None:
        """Initialize weights of the head."""
        super().init_weights()
        kaiming_init(self.convs_all_levels, a=1, distribution='uniform')
        kaiming_init(self.conv_branch, a=1, distribution='uniform')
        kaiming_init(self.conv_pred, a=1, distribution='uniform')
        kaiming_init(self.w1conv, a=1, distribution='uniform')
        kaiming_init(self.w2conv, a=1, distribution='uniform')

    def forward(self, x: Tuple[Tensor]) -> Tensor:
        inputs = x[self.start_level:self.end_level + 1]
        assert len(inputs) == (self.end_level - self.start_level + 1)
        p3 = self.convs_all_levels[0](inputs[0])
        p4 = self.convs_all_levels[1](inputs[1])
        p5 = self.convs_all_levels[2](inputs[2])

        p3_h, p3_w = p3.size()[2:]
        p4_h, p4_w = p4.size()[2:]
        p5_h, p5_w = p5.size()[2:]
        p3_down = self.pool(p3, output_size=(p4_h, p4_w))
        # p4_down = self.pool(p4, output_size=(p5_h, p5_w))
        

        factor_p4_h = p3_h // p4_h
        factor_p4_w = p3_w // p4_w
        factor_p5_h = p3_h // p5_h
        factor_p5_w = p3_w // p5_w
        factor_p_h = p4_h // p5_h
        factor_p_w = p4_w // p5_w
        assert factor_p4_h == factor_p4_w
        assert factor_p5_h == factor_p5_w
        p5u = aligned_bilinear(p5, factor_p_h)
        aa = torch.cat([p3_down, p4, p5u], dim=1)
        aa = self.atten(aa)
        aa = self.attenconv(aa)
        aa = aligned_bilinear(aa, factor_p4_h)
        p4 = aligned_bilinear(p4, factor_p4_h)
        p5 = aligned_bilinear(p5, factor_p5_h)

        
        ff1 = p3 + self.w1conv(p3) * p4 + self.w2conv(p3) * p5
        ff2 = self.w3conv(p3)*aa +aa
        wei1 = F.relu(self.wei1)
        wei2 = F.relu(self.wei2)
        feature_add_all_level = (wei1*ff1+wei2*ff2)/ (wei1 + wei2 + self.eps)

        feature_add_all_level = self.conv_branch(feature_add_all_level)
        feature_pred = self.conv_pred(feature_add_all_level)
        return feature_pred


@MODELS.register_module()
class UWCondInstMaskHead(BaseMaskHead):

    def __init__(self,
                 mask_feature_head: ConfigType,
                 num_layers: int = 3,
                 feat_channels: int = 8,
                 mask_out_stride: int = 4,
                 size_of_interest: int = 8,
                 max_masks_to_train: int = -1,
                 topk_masks_per_img: int = -1,
                 loss_mask: ConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None) -> None:
        super().__init__()
        self.mask_feature_head = UWMaskFeatModule(**mask_feature_head)
        self.mask_feat_stride = self.mask_feature_head.mask_stride
        self.in_channels = self.mask_feature_head.out_channels
        self.num_layers = num_layers
        self.feat_channels = feat_channels
        self.size_of_interest = size_of_interest
        self.mask_out_stride = mask_out_stride
        self.max_masks_to_train = max_masks_to_train
        self.topk_masks_per_img = topk_masks_per_img
        self.prior_generator = MlvlPointGenerator([self.mask_feat_stride])

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.loss_mask = MODELS.build(loss_mask)
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        weight_nums, bias_nums = [], []
        for i in range(self.num_layers):
            if i == 0:
                weight_nums.append((self.in_channels + 2) * self.feat_channels)
                bias_nums.append(self.feat_channels)
            elif i == self.num_layers - 1:
                weight_nums.append(self.feat_channels * 1)
                bias_nums.append(1)
            else:
                weight_nums.append(self.feat_channels * self.feat_channels)
                bias_nums.append(self.feat_channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_params = sum(weight_nums) + sum(bias_nums)

    def parse_dynamic_params(
            self, params: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        """parse the dynamic params for dynamic conv."""
        num_insts = params.size(0)
        params_splits = list(
            torch.split_with_sizes(
                params, self.weight_nums + self.bias_nums, dim=1))
        weight_splits = params_splits[:self.num_layers]
        bias_splits = params_splits[self.num_layers:]
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                weight_splits[i] = weight_splits[i].reshape(
                    num_insts * self.in_channels, -1, 1, 1)
                bias_splits[i] = bias_splits[i].reshape(num_insts *
                                                        self.in_channels)
            else:
                # out_channels x in_channels x 1 x 1
                weight_splits[i] = weight_splits[i].reshape(
                    num_insts * 1, -1, 1, 1)
                bias_splits[i] = bias_splits[i].reshape(num_insts)

        return weight_splits, bias_splits

    def dynamic_conv_forward(self, features: Tensor, weights: List[Tensor],
                             biases: List[Tensor], num_insts: int) -> Tensor:
        """dynamic forward, each layer follow a relu."""
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(x, w, bias=b, stride=1, padding=0, groups=num_insts)
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def forward(self, x: tuple, positive_infos: InstanceList) -> tuple:
        mask_feats = self.mask_feature_head(x)
        return multi_apply(self.forward_single, mask_feats, positive_infos)

    def forward_single(self, mask_feat: Tensor,
                       positive_info: InstanceData) -> Tensor:
        """Forward features of a each image."""
        pos_param_preds = positive_info.get('param_preds')
        pos_points = positive_info.get('points')
        pos_strides = positive_info.get('strides')

        num_inst = pos_param_preds.shape[0]
        mask_feat = mask_feat[None].repeat(num_inst, 1, 1, 1)
        _, _, H, W = mask_feat.size()
        if num_inst == 0:
            return (pos_param_preds.new_zeros((0, 1, H, W)),)

        locations = self.prior_generator.single_level_grid_priors(
            mask_feat.size()[2:], 0, device=mask_feat.device)

        rel_coords = relative_coordinate_maps(locations, pos_points,
                                              pos_strides,
                                              self.size_of_interest,
                                              mask_feat.size()[2:])
        mask_head_inputs = torch.cat([rel_coords, mask_feat], dim=1)
        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)

        weights, biases = self.parse_dynamic_params(pos_param_preds)
        mask_preds = self.dynamic_conv_forward(mask_head_inputs, weights,
                                               biases, num_inst)
        mask_preds = mask_preds.reshape(-1, H, W)
        mask_preds = aligned_bilinear(
            mask_preds.unsqueeze(0),
            int(self.mask_feat_stride / self.mask_out_stride)).squeeze(0)

        return (mask_preds,)

    def loss_by_feat(self, mask_preds: List[Tensor],
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict], positive_infos: InstanceList,
                     **kwargs) -> dict:
        assert positive_infos is not None, \
            'positive_infos should not be None in `CondInstMaskHead`'
        losses = dict()

        loss_mask = 0.
        num_imgs = len(mask_preds)
        total_pos = 0

        for idx in range(num_imgs):
            (mask_pred, pos_mask_targets, num_pos) = \
                self._get_targets_single(
                    mask_preds[idx], batch_gt_instances[idx],
                    positive_infos[idx])
            # mask loss
            total_pos += num_pos
            if num_pos == 0 or pos_mask_targets is None:
                loss = mask_pred.new_zeros(1).mean()
            else:
                loss = self.loss_mask(
                    mask_pred, pos_mask_targets,
                    reduction_override='none').sum()
            loss_mask += loss

        if total_pos == 0:
            total_pos += 1  # avoid nan
        loss_mask = loss_mask / total_pos
        losses.update(loss_mask=loss_mask)
        return losses

    def _get_targets_single(self, mask_preds: Tensor,
                            gt_instances: InstanceData,
                            positive_info: InstanceData):
        gt_bboxes = gt_instances.bboxes
        device = gt_bboxes.device
        gt_masks = gt_instances.masks.to_tensor(
            dtype=torch.bool, device=device).float()

        # process with mask targets
        pos_assigned_gt_inds = positive_info.get('pos_assigned_gt_inds')
        scores = positive_info.get('scores')
        centernesses = positive_info.get('centernesses')
        num_pos = pos_assigned_gt_inds.size(0)

        if gt_masks.size(0) == 0 or num_pos == 0:
            return mask_preds, None, 0
        # Since we're producing (near) full image masks,
        # it'd take too much vram to backprop on every single mask.
        # Thus we select only a subset.
        if (self.max_masks_to_train != -1) and \
                (num_pos > self.max_masks_to_train):
            perm = torch.randperm(num_pos)
            select = perm[:self.max_masks_to_train]
            mask_preds = mask_preds[select]
            pos_assigned_gt_inds = pos_assigned_gt_inds[select]
            num_pos = self.max_masks_to_train
        elif self.topk_masks_per_img != -1:
            unique_gt_inds = pos_assigned_gt_inds.unique()
            num_inst_per_gt = max(
                int(self.topk_masks_per_img / len(unique_gt_inds)), 1)

            keep_mask_preds = []
            keep_pos_assigned_gt_inds = []
            for gt_ind in unique_gt_inds:
                per_inst_pos_inds = (pos_assigned_gt_inds == gt_ind)
                mask_preds_per_inst = mask_preds[per_inst_pos_inds]
                gt_inds_per_inst = pos_assigned_gt_inds[per_inst_pos_inds]
                if sum(per_inst_pos_inds) > num_inst_per_gt:
                    per_inst_scores = scores[per_inst_pos_inds].sigmoid().max(
                        dim=1)[0]
                    per_inst_centerness = centernesses[
                        per_inst_pos_inds].sigmoid().reshape(-1, )
                    select = (per_inst_scores * per_inst_centerness).topk(
                        k=num_inst_per_gt, dim=0)[1]
                    mask_preds_per_inst = mask_preds_per_inst[select]
                    gt_inds_per_inst = gt_inds_per_inst[select]
                keep_mask_preds.append(mask_preds_per_inst)
                keep_pos_assigned_gt_inds.append(gt_inds_per_inst)
            mask_preds = torch.cat(keep_mask_preds)
            pos_assigned_gt_inds = torch.cat(keep_pos_assigned_gt_inds)
            num_pos = pos_assigned_gt_inds.size(0)

        # Follow the origin implement
        start = int(self.mask_out_stride // 2)
        gt_masks = gt_masks[:, start::self.mask_out_stride,
                   start::self.mask_out_stride]
        gt_masks = gt_masks.gt(0.5).float()
        pos_mask_targets = gt_masks[pos_assigned_gt_inds]

        return (mask_preds, pos_mask_targets, num_pos)

    def predict_by_feat(self,
                        mask_preds: List[Tensor],
                        results_list: InstanceList,
                        batch_img_metas: List[dict],
                        rescale: bool = True,
                        **kwargs) -> InstanceList:
        assert len(mask_preds) == len(results_list) == len(batch_img_metas)

        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            results = results_list[img_id]
            bboxes = results.bboxes
            mask_pred = mask_preds[img_id]
            if bboxes.shape[0] == 0 or mask_pred.shape[0] == 0:
                results_list[img_id] = empty_instances(
                    [img_meta],
                    bboxes.device,
                    task_type='mask',
                    instance_results=[results])[0]
            else:
                im_mask = self._predict_by_feat_single(
                    mask_preds=mask_pred,
                    bboxes=bboxes,
                    img_meta=img_meta,
                    rescale=rescale)
                results.masks = im_mask
        return results_list

    def _predict_by_feat_single(self,
                                mask_preds: Tensor,
                                bboxes: Tensor,
                                img_meta: dict,
                                rescale: bool,
                                cfg: OptConfigType = None):
        cfg = self.test_cfg if cfg is None else cfg
        scale_factor = bboxes.new_tensor(img_meta['scale_factor']).repeat(
            (1, 2))
        img_h, img_w = img_meta['img_shape'][:2]
        ori_h, ori_w = img_meta['ori_shape'][:2]

        mask_preds = mask_preds.sigmoid().unsqueeze(0)
        mask_preds = aligned_bilinear(mask_preds, self.mask_out_stride)
        mask_preds = mask_preds[:, :, :img_h, :img_w]
        if rescale:  # in-placed rescale the bboxes
            scale_factor = bboxes.new_tensor(img_meta['scale_factor']).repeat(
                (1, 2))
            bboxes /= scale_factor

            masks = F.interpolate(
                mask_preds, (ori_h, ori_w),
                mode='bilinear',
                align_corners=False).squeeze(0) > cfg.mask_thr
        else:
            masks = mask_preds.squeeze(0) > cfg.mask_thr

        return masks

