import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.core import draw_heatmap_gaussian, gaussian_radius, PseudoSampler
from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.models.builder import HEADS, build_loss
from mmdet3d.models.utils import clip_sigmoid
from mmdet3d.models.fusion_layers import apply_3d_transformation
from mmdet3d.models.model_utils import PositionEmbeddingLearned, TransformerDecoderLayer
from mmdet.core import build_bbox_coder, multi_apply, build_assigner
from mmcv.cnn import ConvModule, build_conv_layer

from .centerpoint_head import SeparateHead


@HEADS.register_module()
class SimpleTransFusionHead(nn.Module):

    def __init__(self,
                 fuse_img=False,
                 fusion_init=False,
                 num_views=6,
                 in_channels_img=64,
                 out_size_factor_img=4,
                 num_proposals=200,
                 nms_kernel_size=3,
                 in_channels=512,
                 hidden_channel=128,
                 num_classes=10,
                 bn_momentum=0.1,
                 # config for Transformer
                 num_heads=8,
                 ffn_channel=256,
                 dropout=0.1,
                 activation='relu',
                 # config for FFN
                 common_heads=dict(),
                 num_heatmap_convs=2,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 bias='auto',
                 # loss
                 iou_pred=False,
                 loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2, alpha=0.25, reduction='mean', loss_weight=1.0),
                 loss_iou=dict(type='VarifocalLoss', use_sigmoid=True, iou_weighted=True, reduction='mean'),
                 loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
                 loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),
                 # others
                 train_cfg=None,
                 test_cfg=None,
                 bbox_coder=None,
            ):
        super(SimpleTransFusionHead, self).__init__()

        self.num_proposals = num_proposals
        self.nms_kernel_size = nms_kernel_size
        self.num_classes = num_classes
        self.bn_momentum = bn_momentum
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.bbox_coder = build_bbox_coder(bbox_coder)

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_heatmap = build_loss(loss_heatmap)

        self.iou_pred = iou_pred
        if self.iou_pred:
            self.loss_iou = build_loss(loss_iou)

        self.shared_conv = build_conv_layer(
            dict(type='Conv2d'),
            in_channels,
            hidden_channel,
            kernel_size=3,
            padding=1,
            bias=bias,
        )

        self.heatmap_head = nn.Sequential(
            ConvModule(
                hidden_channel,
                hidden_channel,
                kernel_size=3,
                padding=1,
                bias=bias,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
            ),
            build_conv_layer(
                dict(type='Conv2d'),
                hidden_channel,
                num_classes,
                kernel_size=3,
                padding=1,
                bias=bias,
            )
        )
        self.class_encoding = nn.Conv1d(num_classes, hidden_channel, 1)

        # transformer decoder layer for object query with LiDAR feature
        self.decoder = nn.ModuleList()
        self.decoder.append(
            TransformerDecoderLayer(
                hidden_channel, num_heads, ffn_channel, dropout, activation,
                self_posembed=PositionEmbeddingLearned(2, hidden_channel),
                cross_posembed=PositionEmbeddingLearned(2, hidden_channel),
            ))

        # prediction Head
        self.prediction_heads = nn.ModuleList()
        heads = copy.deepcopy(common_heads)
        heads.update(dict(heatmap=(self.num_classes, num_heatmap_convs)))
        self.prediction_heads.append(SeparateHead(hidden_channel, heads, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias))

        self.fuse_img = fuse_img
        self.fusion_init = fusion_init
        self.num_views = num_views
        self.out_size_factor_img = out_size_factor_img
        if self.fuse_img:
            self.shared_conv_img = build_conv_layer(
                dict(type='Conv2d'),
                in_channels_img,
                hidden_channel,
                kernel_size=3,
                padding=1,
                bias=bias,
            )

            # transformer decoder layer for img fusion
            self.decoder.append(
                TransformerDecoderLayer(
                    hidden_channel, num_heads, ffn_channel, dropout, activation,
                    self_posembed=PositionEmbeddingLearned(2, hidden_channel),
                    cross_posembed=PositionEmbeddingLearned(2, hidden_channel),
                ))

            if self.fusion_init:
                self.heatmap_head_img = copy.deepcopy(self.heatmap_head)
                for _ in range(num_views):
                    self.decoder.append(
                        TransformerDecoderLayer(
                            hidden_channel, num_heads, ffn_channel, dropout, activation,
                            self_posembed=PositionEmbeddingLearned(2, hidden_channel),
                            cross_posembed=PositionEmbeddingLearned(2, hidden_channel),
                            cross_only=True,
                        ))
                self.fc = nn.Sequential(nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1))
                self.img_feat_collapsed_pos = None

            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(self.num_classes, num_heatmap_convs)))
            self.prediction_heads.append(SeparateHead(hidden_channel * 2, heads, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias))

        self.init_weights()
        self._init_assigner_sampler()

        self.out_size_factor = self.bbox_coder.out_size_factor
        self.voxel_size = self.bbox_coder.voxel_size
        self.pc_range = self.bbox_coder.pc_range

        self.bev_pos = None
        self.img_feat_pos = None

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_y, batch_x = torch.meshgrid(*[torch.linspace(it[0], it[1], it[2]) for it in meshgrid], indexing='ij')
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1) # [1, H*W, 2]
        return coord_base

    def init_weights(self):
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)

        if hasattr(self, 'query'):
            nn.init.xavier_normal_(self.query)

        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def _init_assigner_sampler(self):
        if self.train_cfg is None:
            return
        assert isinstance(self.train_cfg.assigner, dict)
        self.bbox_sampler = PseudoSampler()
        self.bbox_assigner = build_assigner(self.train_cfg.assigner)

    def forward(self, inputs, img_inputs, img_metas):
        inputs = inputs[0]
        if img_inputs is not None:
            img_inputs = img_inputs[0]

        batch_size = inputs.shape[0]
        lidar_feat = self.shared_conv(inputs)
        dense_heatmap = self.heatmap_head(lidar_feat)   # [B, num_classes, H, W]
        heatmap = dense_heatmap.detach().sigmoid()

        lidar_feat_flatten = lidar_feat.view(batch_size, lidar_feat.shape[1], -1)  # [B, C, H*W]

        if self.bev_pos is None:
            x_size = self.test_cfg['grid_size'][0] // self.out_size_factor
            y_size = self.test_cfg['grid_size'][1] // self.out_size_factor
            self.bev_pos = self.create_2D_grid(x_size, y_size).to(lidar_feat.device)

        bev_pos = self.bev_pos.repeat(batch_size, 1, 1)

        if self.fuse_img:
            img_feat = self.shared_conv_img(img_inputs)  # [B * n_views, C, H, W]
            num_channel, img_h, img_w = img_feat.shape[-3:]
            raw_img_feat = img_feat.view(batch_size, self.num_views, num_channel, img_h, img_w)

            if self.fusion_init:
                img_feat = raw_img_feat.permute(0, 2, 3, 1, 4).reshape(batch_size, num_channel, img_h, -1) # [B, C, H, n_views*W]
                img_feat_collapsed = img_feat.max(2).values
                img_feat_collapsed = self.fc(img_feat_collapsed).view(batch_size, num_channel, img_w * self.num_views)

                if self.img_feat_collapsed_pos is None:
                    self.img_feat_collapsed_pos = self.create_2D_grid(1, img_feat_collapsed.shape[-1]).to(img_feat.device)

                bev_feat = lidar_feat_flatten
                for idx_view in range(self.num_views):
                    bev_feat = self.decoder[2 + idx_view](
                        bev_feat, img_feat_collapsed[..., img_w * idx_view: img_w * (idx_view + 1)],
                        bev_pos, self.img_feat_collapsed_pos[:, img_w * idx_view: img_w * (idx_view + 1)])

                dense_heatmap_img = self.heatmap_head_img(bev_feat.view(lidar_feat.shape))  # [B, num_classes, H, W]
                heatmap = (heatmap + dense_heatmap_img.detach().sigmoid()) / 2

        # perform max pooling on heatmap
        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        local_max_inner = F.max_pool2d(heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0)
        local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner
        if self.test_cfg['dataset'] == 'nuScenes':  # for Pedestrian & Traffic_cone in nuScenes
            local_max[:, 8] = F.max_pool2d(heatmap[:, 8], kernel_size=1, stride=1, padding=0)
            local_max[:, 9] = F.max_pool2d(heatmap[:, 9], kernel_size=1, stride=1, padding=0)
        elif self.test_cfg['dataset'] == 'Waymo':   # for Pedestrian & Cyclist in Waymo
            local_max[:, 1] = F.max_pool2d(heatmap[:, 1], kernel_size=1, stride=1, padding=0)
            local_max[:, 2] = F.max_pool2d(heatmap[:, 2], kernel_size=1, stride=1, padding=0)
        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)

        # get top #num_proposals among all classes
        top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[..., :self.num_proposals]
        top_proposals_class = torch.div(top_proposals, heatmap.shape[-1], rounding_mode='floor')
        top_proposals_index = top_proposals % heatmap.shape[-1]
        self.query_labels = top_proposals_class

        # generate query_feat and query_pos
        query_feat = lidar_feat_flatten.gather(index=top_proposals_index[:, None, :].expand(-1, lidar_feat_flatten.shape[1], -1), dim=-1)   # [B, C, K]
        one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(0, 2, 1) # [B, num_classes, K]
        query_cat_encoding = self.class_encoding(one_hot.float())   # [B, C, K]
        query_feat += query_cat_encoding
        query_pos = bev_pos.gather(index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]), dim=1) # [B, K, 2]

        # transformer decoder layer (LiDAR feature as key, value), update query_feat and query_pos
        query_feat = self.decoder[0](query_feat, lidar_feat_flatten, query_pos, bev_pos)
        first_res_layer = self.prediction_heads[0](query_feat)
        first_res_layer['center'] = first_res_layer['center'] + query_pos.permute(0, 2, 1)  # [B, 2, K]
        query_pos = first_res_layer['center'].detach().clone().permute(0, 2, 1) # [B, K, 2]

        # transformer decoder layer (img feature as key, value)
        if self.fuse_img:
            img_feat_flatten = raw_img_feat.view(batch_size, self.num_views, num_channel, -1)  # [B, n_views, C, H*W]
            if self.img_feat_pos is None:
                self.img_feat_pos = self.create_2D_grid(*img_inputs.shape[-2:]).to(img_feat_flatten.device)

            prev_query_feat = query_feat.detach().clone()
            query_feat = torch.zeros_like(query_feat)  # create new container for img query feature
            assert self.voxel_size[0] == self.voxel_size[1] and self.pc_range[0] == self.pc_range[1]    # assert H==W
            query_pos_realmetric = query_pos.permute(0, 2, 1) * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]
            query_pos_3d = torch.cat([query_pos_realmetric, first_res_layer['height']], dim=1).detach().clone() # [B, 3, K]

            vel = copy.deepcopy(first_res_layer['vel'].detach()) if 'vel' in first_res_layer else None
            pred_boxes = self.bbox_coder.decode(
                copy.deepcopy(first_res_layer['heatmap'].detach()),
                copy.deepcopy(first_res_layer['rot'].detach()),
                copy.deepcopy(first_res_layer['dim'].detach()),
                copy.deepcopy(first_res_layer['center'].detach()),
                copy.deepcopy(first_res_layer['height'].detach()),
                vel,
            )

            on_the_image_mask = torch.ones([batch_size, self.num_proposals]).to(query_pos_3d.device) * -1

            for sample_idx in range(batch_size):
                img_meta = img_metas[sample_idx]
                lidar2img_rt = [query_pos_3d.new_tensor(mat) for mat in img_meta['lidar2img']]
                img_scale_factor = query_pos_3d.new_tensor(img_meta.get('scale_factor', [1.0, 1.0])[:2])
                img_crop_offset = query_pos_3d.new_tensor(img_meta.get('img_crop_offset', 0))
                img_flip = img_meta.get('flip', False)

                img_shape = img_meta['img_shape'][:2]
                img_pad_shape = img_meta['input_shape'][:2]
                boxes = LiDARInstance3DBoxes(pred_boxes[sample_idx]['bboxes'][:, :7], box_dim=7)
                query_pos_3d_with_corners = torch.cat([query_pos_3d[sample_idx], boxes.corners.permute(2, 0, 1).view(3, -1)], dim=-1)  # [3, K] + [3, K*8]
                points = query_pos_3d_with_corners.T    # [9K, 3]

                # transform point clouds back to original coordinate system by reverting the data augmentation
                if self.training:
                    points = apply_3d_transformation(points, 'LIDAR', img_meta, reverse=True).detach()

                num_points = points.shape[0]
                for view_idx in range(self.num_views):
                    pts_4d = torch.cat([points, points.new_ones(size=(num_points, 1))], dim=-1)
                    pts_2d = pts_4d @ lidar2img_rt[view_idx].t()

                    pts_2d[:, 2] = torch.clamp(pts_2d[:, 2], min=1e-5)
                    pts_2d[:, 0] /= pts_2d[:, 2]
                    pts_2d[:, 1] /= pts_2d[:, 2]

                    # img transformation: scale -> crop -> flip
                    img_coors = pts_2d[:, 0:2] * img_scale_factor
                    img_coors -= img_crop_offset    # [N, 2]

                    coor_x, coor_y = torch.split(img_coors, 1, dim=1)
                    if img_flip:
                        # by default we take it as horizontal flip, use img_shape before padding for flip
                        orig_h, orig_w = img_shape
                        coor_x = orig_w - coor_x

                    coor_x, coor_corner_x = coor_x[:self.num_proposals, :], coor_x[self.num_proposals:, :]
                    coor_y, coor_corner_y = coor_y[:self.num_proposals, :], coor_y[self.num_proposals:, :]
                    coor_corner_x = coor_corner_x.reshape(self.num_proposals, 8, 1)
                    coor_corner_y = coor_corner_y.reshape(self.num_proposals, 8, 1)
                    coor_corner_xy = torch.cat([coor_corner_x, coor_corner_y], dim=-1)

                    h, w = img_pad_shape
                    on_the_image = (coor_x > 0) * (coor_x < w) * (coor_y > 0) * (coor_y < h)
                    on_the_image = on_the_image.squeeze()
                    # skip the following computation if no object query fall on current image
                    if on_the_image.sum() <= 1:
                        continue
                    on_the_image_mask[sample_idx, on_the_image] = view_idx

                    # add spatial constraint
                    center_ys = (coor_y[on_the_image] / self.out_size_factor_img)
                    center_xs = (coor_x[on_the_image] / self.out_size_factor_img)
                    centers = torch.cat([center_xs, center_ys], dim=-1).int()   # center on the feature map
                    corners = (coor_corner_xy[on_the_image].max(1).values - coor_corner_xy[on_the_image].min(1).values) / self.out_size_factor_img
                    radius = torch.ceil(corners.norm(dim=-1, p=2) / 2).int()    # radius of the minimum circumscribed circle of the wireframe
                    sigma = (radius * 2 + 1) / 6.0
                    distance = (centers[:, None, :] - (self.img_feat_pos - 0.5)).norm(dim=-1) ** 2
                    gaussian_mask = (-distance / (2 * sigma[:, None] ** 2)).exp()
                    gaussian_mask[gaussian_mask < torch.finfo(torch.float32).eps] = 0
                    attn_mask = gaussian_mask

                    query_feat_view = prev_query_feat[sample_idx, :, on_the_image]
                    query_pos_view = torch.cat([center_xs, center_ys], dim=-1)
                    query_feat_view = self.decoder[1](
                        query_feat_view[None], img_feat_flatten[sample_idx: sample_idx + 1, view_idx],
                        query_pos_view[None], self.img_feat_pos, attn_mask=attn_mask.log())
                    query_feat[sample_idx, :, on_the_image] = query_feat_view.clone()

            second_res_layer = self.prediction_heads[1](torch.cat([query_feat, prev_query_feat], dim=1))
            second_res_layer['center'] = second_res_layer['center'] + query_pos.permute(0, 2, 1)  # [B, 2, K]

            self.on_the_image_mask = (on_the_image_mask != -1)
            for key, value in second_res_layer.items():
                pred_dim = value.shape[1]
                valid_img_mask = self.on_the_image_mask.unsqueeze(1).repeat(1, pred_dim, 1)
                first_res_layer[key][valid_img_mask] = second_res_layer[key][valid_img_mask]

        first_res_layer['query_heatmap_score'] = heatmap.gather(index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1), dim=-1)  # [B, num_classes, K]
        first_res_layer['dense_heatmap'] = dense_heatmap_img if self.fuse_img and self.fusion_init else dense_heatmap
        return first_res_layer

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, preds_dict):
        # change preds_dict into list of dict (index by batch_id)
        list_of_pred_dict = []
        for batch_idx in range(len(gt_bboxes_3d)):
            pred_dict = {}
            for key in preds_dict.keys():
                pred_dict[key] = preds_dict[key][batch_idx: batch_idx+1]
            list_of_pred_dict.append(pred_dict)

        res_tuple = multi_apply(self.get_targets_single, gt_bboxes_3d, gt_labels_3d, list_of_pred_dict)

        labels = torch.stack(res_tuple[0])          # [B, K]
        label_weights = torch.stack(res_tuple[1])   # [B, K]
        bbox_targets = torch.stack(res_tuple[2])    # [B, K, code_size]
        bbox_weights = torch.stack(res_tuple[3])    # [B, K, code_size]
        ious = torch.stack(res_tuple[4])            # [B, K]
        num_pos = np.sum(res_tuple[5])
        matched_iou = np.mean(res_tuple[6])
        heatmap = torch.stack(res_tuple[7]) if res_tuple[7][0] is not None else None        # [B, num_classes, H, W]

        return labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_iou, heatmap

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, preds_dict):
        num_proposals = preds_dict['center'].shape[-1]

        # get pred boxes carefully, do not change the network outputs
        score = copy.deepcopy(preds_dict['heatmap'].detach())
        center = copy.deepcopy(preds_dict['center'].detach())
        height = copy.deepcopy(preds_dict['height'].detach())
        dim = copy.deepcopy(preds_dict['dim'].detach())
        rot = copy.deepcopy(preds_dict['rot'].detach())
        vel = copy.deepcopy(preds_dict['vel'].detach()) if 'vel' in preds_dict.keys() else None

        boxes_dict = self.bbox_coder.decode(score, rot, dim, center, height, vel)  # decode the prediction to real world metric bbox
        bboxes_tensor = boxes_dict[0]['bboxes']
        gt_bboxes_tensor = gt_bboxes_3d.tensor.to(score.device)

        assign_result = self.bbox_assigner.assign(bboxes_tensor, gt_bboxes_tensor, gt_labels_3d, score, self.train_cfg)
        sampling_result = self.bbox_sampler.sample(assign_result, bboxes_tensor, gt_bboxes_tensor)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        assert len(pos_inds) + len(neg_inds) == num_proposals

        # create target for loss computation
        bbox_targets = torch.zeros([num_proposals, self.bbox_coder.code_size]).to(center.device)
        bbox_weights = torch.zeros([num_proposals, self.bbox_coder.code_size]).to(center.device)
        ious = assign_result.max_overlaps
        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long) + self.num_classes    # default self.num_classes
        label_weights = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)

        # both pos and neg have classification loss, only pos has regression and iou loss
        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_gt_bboxes)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            labels[pos_inds] = gt_labels_3d[sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # compute dense heatmap targets
        heatmap = None
        if not self.fuse_img or self.fusion_init:
            device = labels.device
            gt_bboxes_3d = torch.cat([gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]], dim=1).to(device)
            grid_size = torch.tensor(self.train_cfg['grid_size'])
            feature_map_size = torch.div(grid_size[:2], self.out_size_factor, rounding_mode='floor')  # [x_len, y_len]
            heatmap = gt_bboxes_3d.new_zeros(self.num_classes, feature_map_size[1], feature_map_size[0])
            for idx in range(len(gt_bboxes_3d)):
                width = gt_bboxes_3d[idx][3]
                length = gt_bboxes_3d[idx][4]
                width = width / self.voxel_size[0] / self.out_size_factor
                length = length / self.voxel_size[1] / self.out_size_factor

                if width > 0 and length > 0:
                    radius = gaussian_radius((length, width), min_overlap=self.train_cfg['gaussian_overlap'])
                    radius = max(self.train_cfg['min_radius'], int(radius))
                    x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]
                    coor_x = (x - self.pc_range[0]) / self.voxel_size[0] / self.out_size_factor
                    coor_y = (y - self.pc_range[1]) / self.voxel_size[1] / self.out_size_factor
                    center = torch.tensor([coor_x, coor_y], dtype=torch.float32, device=device)
                    center_int = center.to(torch.int32)

                    draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]], center_int, radius)

        mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
        return labels, label_weights, bbox_targets, bbox_weights, ious, int(pos_inds.shape[0]), float(mean_iou), heatmap

    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dict, **kwargs):

        labels, label_weights, bbox_targets, bbox_weights, iou_target, num_pos, matched_ious, heatmap = self.get_targets(gt_bboxes_3d, gt_labels_3d, preds_dict)

        if hasattr(self, 'on_the_image_mask'):
            label_weights = label_weights * self.on_the_image_mask
            bbox_weights = bbox_weights * self.on_the_image_mask[:, :, None]
            num_pos = bbox_weights.max(-1).values.sum()

        loss_dict = dict()

        # compute heatmap loss
        if not self.fuse_img or self.fusion_init:
            normalizer = max(heatmap.eq(1).float().sum().item(), 1)
            loss_heatmap = self.loss_heatmap(clip_sigmoid(preds_dict['dense_heatmap']), heatmap, avg_factor=normalizer)
            loss_dict['loss_heatmap'] = loss_heatmap

        # compute classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = preds_dict['heatmap'].permute(0, 2, 1).reshape(-1, self.num_classes)
        loss_cls = self.loss_cls(cls_score, labels, label_weights, avg_factor=max(num_pos, 1))
        loss_dict[f'loss_cls'] = loss_cls

        # compute regression loss
        reg_list = [preds_dict['center'], preds_dict['height'], preds_dict['dim'], preds_dict['rot']]
        if 'vel' in preds_dict.keys():
            reg_list.append(preds_dict['vel'])
        preds = torch.cat(reg_list, dim=1).permute(0, 2, 1)  # [B, num_proposals, code_size]
        code_weights = self.train_cfg.get('code_weights', None)
        reg_weights = bbox_weights * bbox_weights.new_tensor(code_weights)
        loss_bbox = self.loss_bbox(preds, bbox_targets, reg_weights, avg_factor=max(num_pos, 1))
        loss_dict[f'loss_bbox'] = loss_bbox

        # compute iou loss
        if self.iou_pred:
            iou_pred = preds_dict['iou'].squeeze(1)
            loss_iou = self.loss_iou(iou_pred, iou_target, bbox_weights.max(-1).values, avg_factor=max(num_pos, 1))
            loss_dict[f'loss_iou'] = loss_iou

        # add mean iou of batch matched predictions (pos)
        loss_dict[f'matched_ious'] = loss_cls.new_tensor(matched_ious)

        return loss_dict

    def get_bboxes(self, preds_dict, img_metas, img=None, rescale=False):
        batch_score = preds_dict['heatmap'].sigmoid()
        if self.iou_pred:
           batch_score = torch.sqrt(batch_score * preds_dict['iou'].sigmoid())
        one_hot = F.one_hot(self.query_labels, num_classes=self.num_classes).permute(0, 2, 1)
        batch_score = batch_score * preds_dict['query_heatmap_score'] * one_hot

        batch_center = preds_dict['center']
        batch_height = preds_dict['height']
        batch_dim = preds_dict['dim']
        batch_rot = preds_dict['rot']
        batch_vel = preds_dict.get('vel', None)

        ret_layer = self.bbox_coder.decode(batch_score, batch_rot, batch_dim, batch_center, batch_height, batch_vel, filter=True)
        assert len(ret_layer) == 1, "batch size must be 1 during inference"

        res = [[
            img_metas[0]['box_type_3d'](ret_layer[0]['bboxes'], box_dim=ret_layer[0]['bboxes'].shape[-1]),
            ret_layer[0]['scores'],
            ret_layer[0]['labels'].int()
        ]]

        return res
