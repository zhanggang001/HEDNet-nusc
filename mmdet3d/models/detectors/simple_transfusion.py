import torch
import torch.nn.functional as F

from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from .mvx_two_stage import MVXTwoStageDetector


@DETECTORS.register_module()
class SimpleTransFusionDetector(MVXTwoStageDetector):

    def __init__(self, freeze_img=True, **kwargs):
        super(SimpleTransFusionDetector, self).__init__(**kwargs)

        if freeze_img:
            if self.with_img_backbone:
                for param in self.img_backbone.parameters():
                    param.requires_grad = False
            if self.with_img_neck:
                for param in self.img_neck.parameters():
                    param.requires_grad = False

    def extract_pts_feat(self, pts, img_feats, img_metas):
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def forward_train(self, points, img_metas, gt_bboxes_3d, gt_labels_3d, img=None):
        img_feats, pts_feats = self.extract_feat(points, img, img_metas)
        outs = self.pts_bbox_head(pts_feats, img_feats, img_metas)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def simple_test(self, points, img_metas, img=None, rescale=False):
        img_feats, pts_feats = self.extract_feat(points, img, img_metas)
        outs = self.pts_bbox_head(pts_feats, img_feats, img_metas)
        rois = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)
        bbox_results = [bbox3d2result(bboxes, scores, labels) for bboxes, scores, labels in rois]
        return bbox_results


@DETECTORS.register_module()
class SimpleTransFusionDV(SimpleTransFusionDetector):

    @torch.no_grad()
    def voxelize(self, points):
        coors = []
        # dynamic voxelization only provide a coors mapping
        for res in points:
            res_coors = self.pts_voxel_layer(res)
            coors.append(res_coors)
        points = torch.cat(points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch

    def extract_pts_feat(self, points, img_feats, img_metas):
        voxels, coors = self.voxelize(points)   # voxels == concat(points)
        voxel_features, feature_coors = self.pts_voxel_encoder(
            voxels, coors, points, img_feats, img_metas)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, feature_coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x
