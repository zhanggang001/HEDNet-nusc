import copy
import numpy as np
import torch
from mmcv.cnn import ConvModule, build_conv_layer, kaiming_init
from mmcv.runner import force_fp32
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import Linear
from torch.nn.init import xavier_uniform_, constant_

from mmdet3d.core import (circle_nms, draw_heatmap_gaussian, gaussian_radius,
                          xywhr2xyxyr, limit_period, PseudoSampler)
from mmdet3d.core.bbox.structures import rotation_3d_in_axis
from mmdet3d.core import Box3DMode, LiDARInstance3DBoxes
from mmdet3d.models import builder
from mmdet3d.models.builder import HEADS, build_loss
from mmdet3d.models.utils import clip_sigmoid
from mmdet3d.models.fusion_layers import apply_3d_transformation
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu
from mmdet.core import build_bbox_coder, multi_apply, build_assigner, build_sampler, AssignResult
from mmdet3d.ops.roiaware_pool3d import points_in_boxes_batch
from mmdet3d.models.model_utils import PositionEmbeddingLearned, TransformerDecoderLayer


class FFN(nn.Module):
    def __init__(self,
                 in_channels,
                 heads,
                 head_conv=64,
                 final_kernel=1,
                 init_bias=-2.19,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 bias='auto',
                 **kwargs):
        super(FFN, self).__init__()

        self.heads = heads
        self.init_bias = init_bias
        for head in self.heads:
            classes, num_conv = self.heads[head]

            conv_layers = []
            c_in = in_channels
            for i in range(num_conv - 1):
                conv_layers.append(
                    ConvModule(
                        c_in,
                        head_conv,
                        kernel_size=final_kernel,
                        stride=1,
                        padding=final_kernel // 2,
                        bias=bias,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg))
                c_in = head_conv

            conv_layers.append(
                build_conv_layer(
                    conv_cfg,
                    head_conv,
                    classes,
                    kernel_size=final_kernel,
                    stride=1,
                    padding=final_kernel // 2,
                    bias=True))
            conv_layers = nn.Sequential(*conv_layers)

            self.__setattr__(head, conv_layers)

    def init_weights(self):
        """Initialize weights."""
        for head in self.heads:
            if head == 'heatmap':
                self.__getattr__(head)[-1].bias.data.fill_(self.init_bias)
            else:
                for m in self.__getattr__(head).modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_init(m)

    def forward(self, x):
        """Forward function for SepHead.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].

        Returns:
            dict[str: torch.Tensor]: contains the following keys:

                -reg （torch.Tensor): 2D regression value with the \
                    shape of [B, 2, H, W].
                -height (torch.Tensor): Height value with the \
                    shape of [B, 1, H, W].
                -dim (torch.Tensor): Size value with the shape \
                    of [B, 3, H, W].
                -rot (torch.Tensor): Rotation value with the \
                    shape of [B, 1, H, W].
                -vel (torch.Tensor): Velocity value with the \
                    shape of [B, 2, H, W].
                -heatmap (torch.Tensor): Heatmap with the shape of \
                    [B, N, H, W].
        """
        ret_dict = dict()
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict


@HEADS.register_module()
class TransFusionHead(nn.Module):
    def __init__(self,
                 fuse_img=False,
                 num_views=0,
                 in_channels_img=64,
                 out_size_factor_img=4,
                 num_proposals=128,
                 auxiliary=True,
                 in_channels=128 * 3,
                 hidden_channel=128,
                 num_classes=4,
                 # config for Transformer
                 num_decoder_layers=3,
                 num_heads=8,
                 learnable_query_pos=False,
                 initialize_by_heatmap=False,
                 nms_kernel_size=1,
                 ffn_channel=256,
                 dropout=0.1,
                 bn_momentum=0.1,
                 activation='relu',
                 # config for FFN
                 common_heads=dict(),
                 num_heatmap_convs=2,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 bias='auto',
                 # loss
                 loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_iou=dict(type='VarifocalLoss', use_sigmoid=True, iou_weighted=True, reduction='mean'),
                 loss_bbox=dict(type='L1Loss', reduction='mean'),
                 loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean'),
                 # others
                 train_cfg=None,
                 test_cfg=None,
                 bbox_coder=None,
                 ):
        super(TransFusionHead, self).__init__()

        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.auxiliary = auxiliary
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.bn_momentum = bn_momentum
        self.learnable_query_pos = learnable_query_pos
        self.initialize_by_heatmap = initialize_by_heatmap
        self.nms_kernel_size = nms_kernel_size
        if self.initialize_by_heatmap is True:
            assert self.learnable_query_pos is False, "initialized by heatmap is conflicting with learnable query position"
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)
        self.loss_heatmap = build_loss(loss_heatmap)

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.sampling = False

        # a shared convolution
        self.shared_conv = build_conv_layer(
            dict(type='Conv2d'),
            in_channels,
            hidden_channel,
            kernel_size=3,
            padding=1,
            bias=bias,
        )

        if self.initialize_by_heatmap:
            layers = []
            layers.append(ConvModule(
                hidden_channel,
                hidden_channel,
                kernel_size=3,
                padding=1,
                bias=bias,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
            ))
            layers.append(build_conv_layer(
                dict(type='Conv2d'),
                hidden_channel,
                num_classes,
                kernel_size=3,
                padding=1,
                bias=bias,
            ))
            self.heatmap_head = nn.Sequential(*layers)
            self.class_encoding = nn.Conv1d(num_classes, hidden_channel, 1)
        else:
            # query feature
            self.query_feat = nn.Parameter(torch.randn(1, hidden_channel, self.num_proposals))
            self.query_pos = nn.Parameter(torch.rand([1, self.num_proposals, 2]), requires_grad=learnable_query_pos)

        # transformer decoder layers for object query with LiDAR feature
        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(
                    hidden_channel, num_heads, ffn_channel, dropout, activation,
                    self_posembed=PositionEmbeddingLearned(2, hidden_channel),
                    cross_posembed=PositionEmbeddingLearned(2, hidden_channel),
                ))

        # Prediction Head
        self.prediction_heads = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(self.num_classes, num_heatmap_convs)))
            self.prediction_heads.append(FFN(hidden_channel, heads, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias))

        self.fuse_img = fuse_img
        if self.fuse_img:
            self.num_views = num_views
            self.out_size_factor_img = out_size_factor_img
            self.shared_conv_img = build_conv_layer(
                dict(type='Conv2d'),
                in_channels_img,  # channel of img feature map
                hidden_channel,
                kernel_size=3,
                padding=1,
                bias=bias,
            )
            self.heatmap_head_img = copy.deepcopy(self.heatmap_head)
            # transformer decoder layers for img fusion
            self.decoder.append(
                TransformerDecoderLayer(
                    hidden_channel, num_heads, ffn_channel, dropout, activation,
                    self_posembed=PositionEmbeddingLearned(2, hidden_channel),
                    cross_posembed=PositionEmbeddingLearned(2, hidden_channel),
                ))
            # cross-attention only layers for projecting img feature onto BEV
            for i in range(num_views):
                self.decoder.append(
                    TransformerDecoderLayer(
                        hidden_channel, num_heads, ffn_channel, dropout, activation,
                        self_posembed=PositionEmbeddingLearned(2, hidden_channel),
                        cross_posembed=PositionEmbeddingLearned(2, hidden_channel),
                        cross_only=True,
                    ))
            self.fc = nn.Sequential(*[nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1)])

            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(self.num_classes, num_heatmap_convs)))
            self.prediction_heads.append(FFN(hidden_channel * 2, heads, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias))

        self.init_weights()
        self._init_assigner_sampler()

        # Position Embedding for Cross-Attention, which is re-used during training
        x_size = self.test_cfg['grid_size'][0] // self.test_cfg['out_size_factor']
        y_size = self.test_cfg['grid_size'][1] // self.test_cfg['out_size_factor']
        self.bev_pos = self.create_2D_grid(x_size, y_size)

        self.img_feat_pos = None
        self.img_feat_collapsed_pos = None

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_y, batch_x = torch.meshgrid(*[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base

    def init_weights(self):
        # initialize transformer
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        if hasattr(self, 'query'):
            nn.init.xavier_normal_(self.query)
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def _init_assigner_sampler(self):
        """Initialize the target assigner and sampler of the head."""
        if self.train_cfg is None:
            return

        if self.sampling:
            self.bbox_sampler = build_sampler(self.train_cfg.sampler)
        else:
            self.bbox_sampler = PseudoSampler()
        if isinstance(self.train_cfg.assigner, dict):
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
        elif isinstance(self.train_cfg.assigner, list):
            self.bbox_assigner = [
                build_assigner(res) for res in self.train_cfg.assigner
            ]

    def forward_single(self, inputs, img_inputs, img_metas):
        """Forward function for CenterPoint.

        Args:
            inputs (torch.Tensor): Input feature map with the shape of
                [B, 512, 128(H), 128(W)]. (consistent with L748)

        Returns:
            list[dict]: Output results for tasks.
        """
        batch_size = inputs.shape[0]
        lidar_feat = self.shared_conv(inputs)

        #################################
        # image to BEV
        #################################
        lidar_feat_flatten = lidar_feat.view(batch_size, lidar_feat.shape[1], -1)  # [BS, C, H*W]
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)

        if self.fuse_img:
            img_feat = self.shared_conv_img(img_inputs)  # [BS * n_views, C, H, W]

            img_h, img_w, num_channel = img_inputs.shape[-2], img_inputs.shape[-1], img_feat.shape[1]
            raw_img_feat = img_feat.view(batch_size, self.num_views, num_channel, img_h, img_w).permute(0, 2, 3, 1, 4) # [BS, C, H, n_views, W]
            img_feat = raw_img_feat.reshape(batch_size, num_channel, img_h, img_w * self.num_views)  # [BS, C, H, n_views*W]
            img_feat_collapsed = img_feat.max(2).values
            img_feat_collapsed = self.fc(img_feat_collapsed).view(batch_size, num_channel, img_w * self.num_views)

            # positional encoding for image guided query initialization
            if self.img_feat_collapsed_pos is None:
                img_feat_collapsed_pos = self.img_feat_collapsed_pos = self.create_2D_grid(1, img_feat_collapsed.shape[-1]).to(img_feat.device)
            else:
                img_feat_collapsed_pos = self.img_feat_collapsed_pos

            bev_feat = lidar_feat_flatten
            for idx_view in range(self.num_views):
                bev_feat = self.decoder[2 + idx_view](bev_feat, img_feat_collapsed[..., img_w * idx_view:img_w * (idx_view + 1)], bev_pos, img_feat_collapsed_pos[:, img_w * idx_view:img_w * (idx_view + 1)])

        #################################
        # image guided query initialization
        #################################
        if self.initialize_by_heatmap:
            dense_heatmap = self.heatmap_head(lidar_feat)
            dense_heatmap_img = None
            if self.fuse_img:
                dense_heatmap_img = self.heatmap_head_img(bev_feat.view(lidar_feat.shape))  # [BS, num_classes, H, W]
                heatmap = (dense_heatmap.detach().sigmoid() + dense_heatmap_img.detach().sigmoid()) / 2
            else:
                heatmap = dense_heatmap.detach().sigmoid()
            padding = self.nms_kernel_size // 2
            local_max = torch.zeros_like(heatmap)
            # equals to nms radius = voxel_size * out_size_factor * kenel_size
            local_max_inner = F.max_pool2d(heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0)
            local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner
            ## for Pedestrian & Traffic_cone in nuScenes
            if self.test_cfg['dataset'] == 'nuScenes':
                local_max[:, 8, ] = F.max_pool2d(heatmap[:, 8], kernel_size=1, stride=1, padding=0)
                local_max[:, 9, ] = F.max_pool2d(heatmap[:, 9], kernel_size=1, stride=1, padding=0)
            elif self.test_cfg['dataset'] == 'Waymo':  # for Pedestrian & Cyclist in Waymo
                local_max[:, 1, ] = F.max_pool2d(heatmap[:, 1], kernel_size=1, stride=1, padding=0)
                local_max[:, 2, ] = F.max_pool2d(heatmap[:, 2], kernel_size=1, stride=1, padding=0)
            heatmap = heatmap * (heatmap == local_max)
            heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)

            # top #num_proposals among all classes
            top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[..., :self.num_proposals]
            top_proposals_class = top_proposals // heatmap.shape[-1]
            top_proposals_index = top_proposals % heatmap.shape[-1]
            query_feat = lidar_feat_flatten.gather(index=top_proposals_index[:, None, :].expand(-1, lidar_feat_flatten.shape[1], -1), dim=-1)
            self.query_labels = top_proposals_class

            # add category embedding
            one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(0, 2, 1)
            query_cat_encoding = self.class_encoding(one_hot.float())
            query_feat += query_cat_encoding

            query_pos = bev_pos.gather(index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]), dim=1)
        else:
            query_feat = self.query_feat.repeat(batch_size, 1, 1)  # [BS, C, num_proposals]
            base_xyz = self.query_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)  # [BS, num_proposals, 2]

        #################################
        # transformer decoder layer (LiDAR feature as K,V)
        #################################
        ret_dicts = []
        for i in range(self.num_decoder_layers):
            prefix = 'last_' if (i == self.num_decoder_layers - 1) else f'{i}head_'

            # Transformer Decoder Layer
            # :param query: B C Pq    :param query_pos: B Pq 3/6
            query_feat = self.decoder[i](query_feat, lidar_feat_flatten, query_pos, bev_pos)

            # Prediction
            res_layer = self.prediction_heads[i](query_feat)
            res_layer['center'] = res_layer['center'] + query_pos.permute(0, 2, 1)
            first_res_layer = res_layer
            if not self.fuse_img:
                ret_dicts.append(res_layer)

            # for next level positional embedding
            query_pos = res_layer['center'].detach().clone().permute(0, 2, 1)

        #################################
        # transformer decoder layer (img feature as K,V)
        #################################
        if self.fuse_img:
            # positional encoding for image fusion
            img_feat = raw_img_feat.permute(0, 3, 1, 2, 4) # [BS, n_views, C, H, W]
            img_feat_flatten = img_feat.view(batch_size, self.num_views, num_channel, -1)  # [BS, n_views, C, H*W]
            if self.img_feat_pos is None:
                (h, w) = img_inputs.shape[-2], img_inputs.shape[-1]
                img_feat_pos = self.img_feat_pos = self.create_2D_grid(h, w).to(img_feat_flatten.device)
            else:
                img_feat_pos = self.img_feat_pos

            prev_query_feat = query_feat.detach().clone()
            query_feat = torch.zeros_like(query_feat)  # create new container for img query feature
            query_pos_realmetric = query_pos.permute(0, 2, 1) * self.test_cfg['out_size_factor'] * self.test_cfg['voxel_size'][0] + self.test_cfg['pc_range'][0]
            query_pos_3d = torch.cat([query_pos_realmetric, res_layer['height']], dim=1).detach().clone()
            if 'vel' in res_layer:
                vel = copy.deepcopy(res_layer['vel'].detach())
            else:
                vel = None
            pred_boxes = self.bbox_coder.decode(
                copy.deepcopy(res_layer['heatmap'].detach()),
                copy.deepcopy(res_layer['rot'].detach()),
                copy.deepcopy(res_layer['dim'].detach()),
                copy.deepcopy(res_layer['center'].detach()),
                copy.deepcopy(res_layer['height'].detach()),
                vel,
            )

            on_the_image_mask = torch.ones([batch_size, self.num_proposals]).to(query_pos_3d.device) * -1

            for sample_idx in range(batch_size if self.fuse_img else 0):
                lidar2img_rt = query_pos_3d.new_tensor(img_metas[sample_idx]['lidar2img'])
                img_scale_factor = (
                    query_pos_3d.new_tensor(img_metas[sample_idx]['scale_factor'][:2]
                                            if 'scale_factor' in img_metas[sample_idx].keys() else [1.0, 1.0]))
                img_flip = img_metas[sample_idx]['flip'] if 'flip' in img_metas[sample_idx].keys() else False
                img_crop_offset = (
                    query_pos_3d.new_tensor(img_metas[sample_idx]['img_crop_offset'])
                    if 'img_crop_offset' in img_metas[sample_idx].keys() else 0)
                img_shape = img_metas[sample_idx]['img_shape'][:2]
                img_pad_shape = img_metas[sample_idx]['input_shape'][:2]
                boxes = LiDARInstance3DBoxes(pred_boxes[sample_idx]['bboxes'][:, :7], box_dim=7)
                query_pos_3d_with_corners = torch.cat([query_pos_3d[sample_idx], boxes.corners.permute(2, 0, 1).view(3, -1)], dim=-1)  # [3, num_proposals] + [3, num_proposals*8]
                # transform point clouds back to original coordinate system by reverting the data augmentation
                if batch_size == 1:  # skip during inference to save time
                    points = query_pos_3d_with_corners.T
                else:
                    points = apply_3d_transformation(query_pos_3d_with_corners.T, 'LIDAR', img_metas[sample_idx], reverse=True).detach()
                num_points = points.shape[0]

                for view_idx in range(self.num_views):
                    pts_4d = torch.cat([points, points.new_ones(size=(num_points, 1))], dim=-1)
                    pts_2d = pts_4d @ lidar2img_rt[view_idx].t()

                    pts_2d[:, 2] = torch.clamp(pts_2d[:, 2], min=1e-5)
                    pts_2d[:, 0] /= pts_2d[:, 2]
                    pts_2d[:, 1] /= pts_2d[:, 2]

                    # img transformation: scale -> crop -> flip
                    # the image is resized by img_scale_factor
                    img_coors = pts_2d[:, 0:2] * img_scale_factor  # Nx2
                    img_coors -= img_crop_offset

                    # grid sample, the valid grid range should be in [-1,1]
                    coor_x, coor_y = torch.split(img_coors, 1, dim=1)  # each is Nx1

                    if img_flip:
                        # by default we take it as horizontal flip
                        # use img_shape before padding for flip
                        orig_h, orig_w = img_shape
                        coor_x = orig_w - coor_x

                    coor_x, coor_corner_x = coor_x[0:self.num_proposals, :], coor_x[self.num_proposals:, :]
                    coor_y, coor_corner_y = coor_y[0:self.num_proposals, :], coor_y[self.num_proposals:, :]
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
                    centers = torch.cat([center_xs, center_ys], dim=-1).int()  # center on the feature map
                    corners = (coor_corner_xy[on_the_image].max(1).values - coor_corner_xy[on_the_image].min(1).values) / self.out_size_factor_img
                    radius = torch.ceil(corners.norm(dim=-1, p=2) / 2).int()  # radius of the minimum circumscribed circle of the wireframe
                    sigma = (radius * 2 + 1) / 6.0
                    distance = (centers[:, None, :] - (img_feat_pos - 0.5)).norm(dim=-1) ** 2
                    gaussian_mask = (-distance / (2 * sigma[:, None] ** 2)).exp()
                    gaussian_mask[gaussian_mask < torch.finfo(torch.float32).eps] = 0
                    attn_mask = gaussian_mask

                    query_feat_view = prev_query_feat[sample_idx, :, on_the_image]
                    query_pos_view = torch.cat([center_xs, center_ys], dim=-1)
                    query_feat_view = self.decoder[self.num_decoder_layers](query_feat_view[None], img_feat_flatten[sample_idx:sample_idx + 1, view_idx], query_pos_view[None], img_feat_pos, attn_mask=attn_mask.log())
                    query_feat[sample_idx, :, on_the_image] = query_feat_view.clone()

            self.on_the_image_mask = (on_the_image_mask != -1)
            res_layer = self.prediction_heads[self.num_decoder_layers](torch.cat([query_feat, prev_query_feat], dim=1))
            res_layer['center'] = res_layer['center'] + query_pos.permute(0, 2, 1)
            for key, value in res_layer.items():
                pred_dim = value.shape[1]
                res_layer[key][~self.on_the_image_mask.unsqueeze(1).repeat(1, pred_dim, 1)] = first_res_layer[key][~self.on_the_image_mask.unsqueeze(1).repeat(1, pred_dim, 1)]
            ret_dicts.append(res_layer)

        if self.initialize_by_heatmap:
            ret_dicts[0]['query_heatmap_score'] = heatmap.gather(index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1), dim=-1)  # [bs, num_classes, num_proposals]
            if self.fuse_img:
                ret_dicts[0]['dense_heatmap'] = dense_heatmap_img
            else:
                ret_dicts[0]['dense_heatmap'] = dense_heatmap

        if self.auxiliary is False:
            # only return the results of last decoder layer
            return [ret_dicts[-1]]

        # return all the layer's results for auxiliary superivison
        new_res = {}
        for key in ret_dicts[0].keys():
            if key not in ['dense_heatmap', 'dense_heatmap_old', 'query_heatmap_score']:
                new_res[key] = torch.cat([ret_dict[key] for ret_dict in ret_dicts], dim=-1)
            else:
                new_res[key] = ret_dicts[0][key]
        return [new_res]

    def forward(self, feats, img_feats, img_metas):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.

        Returns:
            tuple(list[dict]): Output results. first index by level, second index by layer
        """
        if img_feats is None:
            img_feats = [None]
        res = multi_apply(self.forward_single, feats, img_feats, [img_metas])
        assert len(res) == 1, "only support one level features."
        return res

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, preds_dict):
        """Generate training targets.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.
            preds_dicts (tuple of dict): first index by layer (default 1)
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.

                - torch.Tensor: classification target.  [BS, num_proposals]
                - torch.Tensor: classification weights (mask)  [BS, num_proposals]
                - torch.Tensor: regression target. [BS, num_proposals, 8]
                - torch.Tensor: regression weights. [BS, num_proposals, 8]
        """
        # change preds_dict into list of dict (index by batch_id)
        # preds_dict[0]['center'].shape [bs, 3, num_proposal]
        list_of_pred_dict = []
        for batch_idx in range(len(gt_bboxes_3d)):
            pred_dict = {}
            for key in preds_dict[0].keys():
                pred_dict[key] = preds_dict[0][key][batch_idx:batch_idx + 1]
            list_of_pred_dict.append(pred_dict)

        assert len(gt_bboxes_3d) == len(list_of_pred_dict)

        res_tuple = multi_apply(self.get_targets_single, gt_bboxes_3d, gt_labels_3d, list_of_pred_dict, np.arange(len(gt_labels_3d)))
        labels = torch.cat(res_tuple[0], dim=0)
        label_weights = torch.cat(res_tuple[1], dim=0)
        bbox_targets = torch.cat(res_tuple[2], dim=0)
        bbox_weights = torch.cat(res_tuple[3], dim=0)
        ious = torch.cat(res_tuple[4], dim=0)
        num_pos = np.sum(res_tuple[5])
        matched_ious = np.mean(res_tuple[6])
        if self.initialize_by_heatmap:
            heatmap = torch.cat(res_tuple[7], dim=0)
            return labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious, heatmap
        else:
            return labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, preds_dict, batch_idx):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.
            preds_dict (dict): dict of prediction result for a single sample
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.

                - torch.Tensor: classification target.  [1, num_proposals]
                - torch.Tensor: classification weights (mask)  [1, num_proposals]
                - torch.Tensor: regression target. [1, num_proposals, 8]
                - torch.Tensor: regression weights. [1, num_proposals, 8]
                - torch.Tensor: iou target. [1, num_proposals]
                - int: number of positive proposals
        """
        num_proposals = preds_dict['center'].shape[-1]

        # get pred boxes, carefully ! donot change the network outputs
        score = copy.deepcopy(preds_dict['heatmap'].detach())
        center = copy.deepcopy(preds_dict['center'].detach())
        height = copy.deepcopy(preds_dict['height'].detach())
        dim = copy.deepcopy(preds_dict['dim'].detach())
        rot = copy.deepcopy(preds_dict['rot'].detach())
        if 'vel' in preds_dict.keys():
            vel = copy.deepcopy(preds_dict['vel'].detach())
        else:
            vel = None

        boxes_dict = self.bbox_coder.decode(score, rot, dim, center, height, vel)  # decode the prediction to real world metric bbox
        bboxes_tensor = boxes_dict[0]['bboxes']
        gt_bboxes_tensor = gt_bboxes_3d.tensor.to(score.device)
        # each layer should do label assign seperately.
        if self.auxiliary:
            num_layer = self.num_decoder_layers
        else:
            num_layer = 1

        assign_result_list = []
        for idx_layer in range(num_layer):
            bboxes_tensor_layer = bboxes_tensor[self.num_proposals * idx_layer:self.num_proposals * (idx_layer + 1), :]
            score_layer = score[..., self.num_proposals * idx_layer:self.num_proposals * (idx_layer + 1)]

            if self.train_cfg.assigner.type == 'HungarianAssigner3D':
                assign_result = self.bbox_assigner.assign(bboxes_tensor_layer, gt_bboxes_tensor, gt_labels_3d, score_layer, self.train_cfg)
            elif self.train_cfg.assigner.type == 'HeuristicAssigner':
                assign_result = self.bbox_assigner.assign(bboxes_tensor_layer, gt_bboxes_tensor, None, gt_labels_3d, self.query_labels[batch_idx])
            else:
                raise NotImplementedError
            assign_result_list.append(assign_result)

        # combine assign result of each layer
        assign_result_ensemble = AssignResult(
            num_gts=sum([res.num_gts for res in assign_result_list]),
            gt_inds=torch.cat([res.gt_inds for res in assign_result_list]),
            max_overlaps=torch.cat([res.max_overlaps for res in assign_result_list]),
            labels=torch.cat([res.labels for res in assign_result_list]),
        )
        sampling_result = self.bbox_sampler.sample(assign_result_ensemble, bboxes_tensor, gt_bboxes_tensor)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        assert len(pos_inds) + len(neg_inds) == num_proposals

        # create target for loss computation
        bbox_targets = torch.zeros([num_proposals, self.bbox_coder.code_size]).to(center.device)
        bbox_weights = torch.zeros([num_proposals, self.bbox_coder.code_size]).to(center.device)
        ious = assign_result_ensemble.max_overlaps
        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)
        label_weights = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)

        if gt_labels_3d is not None:  # default label is -1
            labels += self.num_classes

        # both pos and neg have classification loss, only pos has regression and iou loss
        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_gt_bboxes)

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels_3d is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels_3d[sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # # compute dense heatmap targets
        if self.initialize_by_heatmap:
            device = labels.device
            gt_bboxes_3d = torch.cat([gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]], dim=1).to(device)
            grid_size = torch.tensor(self.train_cfg['grid_size'])
            pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
            voxel_size = torch.tensor(self.train_cfg['voxel_size'])
            feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']  # [x_len, y_len]
            heatmap = gt_bboxes_3d.new_zeros(self.num_classes, feature_map_size[1], feature_map_size[0])
            for idx in range(len(gt_bboxes_3d)):
                width = gt_bboxes_3d[idx][3]
                length = gt_bboxes_3d[idx][4]
                width = width / voxel_size[0] / self.train_cfg['out_size_factor']
                length = length / voxel_size[1] / self.train_cfg['out_size_factor']
                if width > 0 and length > 0:
                    radius = gaussian_radius((length, width), min_overlap=self.train_cfg['gaussian_overlap'])
                    radius = max(self.train_cfg['min_radius'], int(radius))
                    x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]

                    coor_x = (x - pc_range[0]) / voxel_size[0] / self.train_cfg['out_size_factor']
                    coor_y = (y - pc_range[1]) / voxel_size[1] / self.train_cfg['out_size_factor']

                    center = torch.tensor([coor_x, coor_y], dtype=torch.float32, device=device)
                    center_int = center.to(torch.int32)
                    draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]], center_int, radius)

            mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
            return labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], ious[None], int(pos_inds.shape[0]), float(mean_iou), heatmap[None]

        else:
            mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
            return labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], ious[None], int(pos_inds.shape[0]), float(mean_iou)

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """Loss function for CenterHead.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (list[list[dict]]): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        if self.initialize_by_heatmap:
            labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious, heatmap = self.get_targets(gt_bboxes_3d, gt_labels_3d, preds_dicts[0])
        else:
            labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious = self.get_targets(gt_bboxes_3d, gt_labels_3d, preds_dicts[0])
        if hasattr(self, 'on_the_image_mask'):
            label_weights = label_weights * self.on_the_image_mask
            bbox_weights = bbox_weights * self.on_the_image_mask[:, :, None]
            num_pos = bbox_weights.max(-1).values.sum()
        preds_dict = preds_dicts[0][0]
        loss_dict = dict()

        if self.initialize_by_heatmap:
            # compute heatmap loss
            loss_heatmap = self.loss_heatmap(clip_sigmoid(preds_dict['dense_heatmap']), heatmap, avg_factor=max(heatmap.eq(1).float().sum().item(), 1))
            loss_dict['loss_heatmap'] = loss_heatmap

        # compute loss for each layer
        for idx_layer in range(self.num_decoder_layers if self.auxiliary else 1):
            if idx_layer == self.num_decoder_layers - 1 or (idx_layer == 0 and self.auxiliary is False):
                prefix = 'layer_-1'
            else:
                prefix = f'layer_{idx_layer}'

            layer_labels = labels[..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals].reshape(-1)
            layer_label_weights = label_weights[..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals].reshape(-1)
            layer_score = preds_dict['heatmap'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
            layer_cls_score = layer_score.permute(0, 2, 1).reshape(-1, self.num_classes)
            layer_loss_cls = self.loss_cls(layer_cls_score, layer_labels, layer_label_weights, avg_factor=max(num_pos, 1))

            layer_center = preds_dict['center'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
            layer_height = preds_dict['height'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
            layer_rot = preds_dict['rot'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
            layer_dim = preds_dict['dim'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
            preds = torch.cat([layer_center, layer_height, layer_dim, layer_rot], dim=1).permute(0, 2, 1)  # [BS, num_proposals, code_size]
            if 'vel' in preds_dict.keys():
                layer_vel = preds_dict['vel'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
                preds = torch.cat([layer_center, layer_height, layer_dim, layer_rot, layer_vel], dim=1).permute(0, 2, 1)  # [BS, num_proposals, code_size]
            code_weights = self.train_cfg.get('code_weights', None)
            layer_bbox_weights = bbox_weights[:, idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals, :]
            layer_reg_weights = layer_bbox_weights * layer_bbox_weights.new_tensor(code_weights)
            layer_bbox_targets = bbox_targets[:, idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals, :]
            layer_loss_bbox = self.loss_bbox(preds, layer_bbox_targets, layer_reg_weights, avg_factor=max(num_pos, 1))

            # layer_iou = preds_dict['iou'][..., idx_layer*self.num_proposals:(idx_layer+1)*self.num_proposals].squeeze(1)
            # layer_iou_target = ious[..., idx_layer*self.num_proposals:(idx_layer+1)*self.num_proposals]
            # layer_loss_iou = self.loss_iou(layer_iou, layer_iou_target, layer_bbox_weights.max(-1).values, avg_factor=max(num_pos, 1))

            loss_dict[f'{prefix}_loss_cls'] = layer_loss_cls
            loss_dict[f'{prefix}_loss_bbox'] = layer_loss_bbox
            # loss_dict[f'{prefix}_loss_iou'] = layer_loss_iou

        loss_dict[f'matched_ious'] = layer_loss_cls.new_tensor(matched_ious)

        return loss_dict

    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False, for_roi=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.

        Returns:
            list[list[dict]]: Decoded bbox, scores and labels for each layer & each batch
        """
        rets = []
        for layer_id, preds_dict in enumerate(preds_dicts):
            batch_size = preds_dict[0]['heatmap'].shape[0]
            batch_score = preds_dict[0]['heatmap'][..., -self.num_proposals:].sigmoid()
            # if self.loss_iou.loss_weight != 0:
            #    batch_score = torch.sqrt(batch_score * preds_dict[0]['iou'][..., -self.num_proposals:].sigmoid())
            one_hot = F.one_hot(self.query_labels, num_classes=self.num_classes).permute(0, 2, 1)
            batch_score = batch_score * preds_dict[0]['query_heatmap_score'] * one_hot

            batch_center = preds_dict[0]['center'][..., -self.num_proposals:]
            batch_height = preds_dict[0]['height'][..., -self.num_proposals:]
            batch_dim = preds_dict[0]['dim'][..., -self.num_proposals:]
            batch_rot = preds_dict[0]['rot'][..., -self.num_proposals:]
            batch_vel = None
            if 'vel' in preds_dict[0]:
                batch_vel = preds_dict[0]['vel'][..., -self.num_proposals:]

            temp = self.bbox_coder.decode(batch_score, batch_rot, batch_dim, batch_center, batch_height, batch_vel, filter=True)

            if self.test_cfg['dataset'] == 'nuScenes':
                self.tasks = [
                    dict(num_class=8, class_names=[], indices=[0, 1, 2, 3, 4, 5, 6, 7], radius=-1),
                    dict(num_class=1, class_names=['pedestrian'], indices=[8], radius=0.175),
                    dict(num_class=1, class_names=['traffic_cone'], indices=[9], radius=0.175),
                ]
            elif self.test_cfg['dataset'] == 'Waymo':
                self.tasks = [
                    dict(num_class=1, class_names=['Car'], indices=[0], radius=0.7),
                    dict(num_class=1, class_names=['Pedestrian'], indices=[1], radius=0.7),
                    dict(num_class=1, class_names=['Cyclist'], indices=[2], radius=0.7),
                ]

            ret_layer = []
            for i in range(batch_size):
                boxes3d = temp[i]['bboxes']
                scores = temp[i]['scores']
                labels = temp[i]['labels']
                ## adopt circle nms for different categories
                if self.test_cfg['nms_type'] != None:
                    keep_mask = torch.zeros_like(scores)
                    for task in self.tasks:
                        task_mask = torch.zeros_like(scores)
                        for cls_idx in task['indices']:
                            task_mask += labels == cls_idx
                        task_mask = task_mask.bool()
                        if task['radius'] > 0:
                            if self.test_cfg['nms_type'] == 'circle':
                                boxes_for_nms = torch.cat([boxes3d[task_mask][:, :2], scores[:, None][task_mask]], dim=1)
                                task_keep_indices = torch.tensor(
                                    circle_nms(
                                        boxes_for_nms.detach().cpu().numpy(),
                                        task['radius'],
                                    )
                                )
                            else:
                                boxes_for_nms = xywhr2xyxyr(img_metas[i]['box_type_3d'](boxes3d[task_mask][:, :7], 7).bev)
                                top_scores = scores[task_mask]
                                task_keep_indices = nms_gpu(
                                    boxes_for_nms,
                                    top_scores,
                                    thresh=task['radius'],
                                    pre_maxsize=self.test_cfg['pre_maxsize'],
                                    post_max_size=self.test_cfg['post_maxsize'],
                                )
                        else:
                            task_keep_indices = torch.arange(task_mask.sum())
                        if task_keep_indices.shape[0] != 0:
                            keep_indices = torch.where(task_mask != 0)[0][task_keep_indices]
                            keep_mask[keep_indices] = 1
                    keep_mask = keep_mask.bool()
                    ret = dict(bboxes=boxes3d[keep_mask], scores=scores[keep_mask], labels=labels[keep_mask])
                else:  # no nms
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                ret_layer.append(ret)
            rets.append(ret_layer)
        assert len(rets) == 1
        assert len(rets[0]) == 1
        res = [[
            img_metas[0]['box_type_3d'](rets[0][0]['bboxes'], box_dim=rets[0][0]['bboxes'].shape[-1]),
            rets[0][0]['scores'],
            rets[0][0]['labels'].int()
        ]]
        return res
