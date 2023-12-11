from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import load_checkpoint
import torch.nn as nn

from mmdet.models import BACKBONES


@BACKBONES.register_module()
class SECOND(nn.Module):
    """Backbone network for SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of each stage.
        norm_cfg (dict): Config dict of normalization layers.
        conv_cfg (dict): Config dict of convolutional layers.
    """

    def __init__(self,
                 in_channels=128,
                 out_channels=[128, 128, 256],
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 conv_cfg=dict(type='Conv2d', bias=False)):
        super(SECOND, self).__init__()
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        in_filters = [in_channels, *out_channels[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = []
        for i, layer_num in enumerate(layer_nums):
            block = [
                build_conv_layer(
                    conv_cfg,
                    in_filters[i],
                    out_channels[i],
                    3,
                    stride=layer_strides[i],
                    padding=1),
                build_norm_layer(norm_cfg, out_channels[i])[1],
                nn.ReLU(inplace=True),
            ]
            for j in range(layer_num):
                block.append(
                    build_conv_layer(
                        conv_cfg,
                        out_channels[i],
                        out_channels[i],
                        3,
                        padding=1))
                block.append(build_norm_layer(norm_cfg, out_channels[i])[1])
                block.append(nn.ReLU(inplace=True))

            block = nn.Sequential(*block)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

    def init_weights(self, pretrained=None):
        """Initialize weights of the 2D backbone."""
        # Do not initialize the conv layers
        # to follow the original implementation
        if isinstance(pretrained, str):
            from mmdet3d.utils import get_root_logger
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        outs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            outs.append(x)
        return tuple(outs)


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, padding=1, downsample=False):
        super().__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu2 = nn.ReLU()

        self.downsample = downsample
        if self.downsample:
            self.downsample_layer = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample_layer(x)

        out += identity
        out = self.relu2(out)
        return out


class DEDBackbone(nn.Module):

    def __init__(self, model_cfg, input_channels):
        super().__init__()

        num_SBB = model_cfg['NUM_SBB']
        down_strides = model_cfg['DOWN_STRIDES']
        dim = model_cfg['FEATURE_DIM']
        assert len(num_SBB) == len(down_strides)

        num_levels = len(down_strides)

        first_block = []
        if input_channels != dim:
            first_block.append(BasicBlock(input_channels, dim, down_strides[0], 1, True))
        first_block += [BasicBlock(dim, dim) for _ in range(num_SBB[0])]
        self.encoder = nn.ModuleList([nn.Sequential(*first_block)])

        for idx in range(1, num_levels):
            cur_layers = [BasicBlock(dim, dim, down_strides[idx], 1, True)]
            cur_layers.extend([BasicBlock(dim, dim) for _ in range(num_SBB[idx])])
            self.encoder.append(nn.Sequential(*cur_layers))

        self.decoder = nn.ModuleList()
        self.decoder_norm = nn.ModuleList()
        for idx in range(num_levels - 1, 0, -1):
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(dim, dim, down_strides[idx], down_strides[idx], bias=False),
                    nn.BatchNorm2d(dim, eps=1e-3, momentum=0.01),
                    nn.ReLU()
                )
            )
            self.decoder_norm.append(nn.BatchNorm2d(dim, eps=1e-3, momentum=0.01))

        self.num_bev_features = dim
        self.init_weights()

    def init_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.encoder[0](x)

        feats = [x]
        for conv in self.encoder[1:]:
            x = conv(x)
            feats.append(x)

        for deconv, norm, up_x in zip(self.decoder, self.decoder_norm, feats[:-1][::-1]):
            x = norm(deconv(x) + up_x)
        return x


@BACKBONES.register_module()
class CascadeDEDBackbone(nn.Module):

    def __init__(self, model_cfg, in_channels):
        super().__init__()

        self.layers = nn.ModuleList()
        for idx in range(model_cfg['NUM_LAYERS']):
            input_dim = in_channels if idx == 0 else model_cfg['FEATURE_DIM']
            self.layers.append(DEDBackbone(model_cfg, input_dim))

        self.num_bev_features = self.layers[0].num_bev_features

    def init_weights(self, pretrained=None):
        pass

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return [x]
