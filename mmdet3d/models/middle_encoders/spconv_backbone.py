from functools import partial
from torch import nn as nn

from mmdet3d.ops import spconv as spconv
from ..registry import MIDDLE_ENCODERS


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0, conv_type='subm', norm_fn=None):
    assert conv_type in ['subm', 'spconv', 'inverseconv']

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(
            in_channels, out_channels, kernel_size, padding=padding, bias=False, indice_key=indice_key)

    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False, indice_key=indice_key)

    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(
            in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)

    return spconv.SparseSequential(conv, norm_fn(out_channels), nn.ReLU())


class SparseBasicBlock(spconv.SparseModule):

    def __init__(self, inplanes, planes, norm_fn, indice_key):
        super(SparseBasicBlock, self).__init__()

        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=1, padding=1, bias=True, indice_key=indice_key)
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()

        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=True, indice_key=indice_key)
        self.bn2 = norm_fn(planes)

    def forward(self, x):
        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))
        out = out.replace_feature(self.relu(out.features + x.features))
        return out


class SEDBlock(spconv.SparseModule):

    def __init__(self, dim, kernel_size, stride, num_SBB, norm_fn, indice_key):
        super(SEDBlock, self).__init__()

        first_block = post_act_block(
            dim, dim, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
            norm_fn=norm_fn, indice_key=f'spconv_{indice_key}', conv_type='spconv')

        block_list = [first_block if stride > 1 else nn.Identity()]
        for _ in range(num_SBB):
            block_list.append(
                SparseBasicBlock(dim, dim, norm_fn=norm_fn, indice_key=indice_key))

        self.blocks = spconv.SparseSequential(*block_list)

    def forward(self, x):
        return self.blocks(x)


class SEDLayer(spconv.SparseModule):

    def __init__(self, dim: int, down_kernel_size: list, down_stride: list, num_SBB: list, norm_fn, indice_key):
        super().__init__()

        assert down_stride[0] == 1 # hard code
        assert len(down_kernel_size) == len(down_stride) == len(num_SBB)

        self.encoder = nn.ModuleList()
        for idx in range(len(down_stride)):
            self.encoder.append(
                SEDBlock(dim, down_kernel_size[idx], down_stride[idx], num_SBB[idx], norm_fn, f"{indice_key}_{idx}"))

        downsample_times = len(down_stride[1:])
        self.decoder = nn.ModuleList()
        self.decoder_norm = nn.ModuleList()
        for idx, kernel_size in enumerate(down_kernel_size[1:]):
            self.decoder.append(
                post_act_block(
                    dim, dim, kernel_size, norm_fn=norm_fn, conv_type='inverseconv',
                    indice_key=f'spconv_{indice_key}_{downsample_times - idx}'))
            self.decoder_norm.append(norm_fn(dim))

    def forward(self, x):
        features = []
        for conv in self.encoder:
            x = conv(x)
            features.append(x)

        x = features[-1]
        for deconv, norm, up_x in zip(self.decoder, self.decoder_norm, features[:-1][::-1]):
            x = deconv(x)
            x = x.replace_feature(x.features + up_x.features)
            x = x.replace_feature(norm(x.features))
        return x


@MIDDLE_ENCODERS.register_module()
class HEDNet(nn.Module):

    def __init__(self, model_cfg, in_channels, sparse_shape, **kwargs):
        super().__init__()

        self.sparse_shape = sparse_shape
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        dim = model_cfg['FEATURE_DIM']
        num_layers = model_cfg['NUM_LAYERS']
        num_SBB = model_cfg['NUM_SBB']
        down_kernel_size = model_cfg['DOWN_KERNEL_SIZE']
        down_stride = model_cfg['DOWN_STRIDE']

        # [1888, 1888, 41] -> [944, 944, 21]
        self.conv1 = spconv.SparseSequential(
            post_act_block(in_channels, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1', conv_type='subm'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='stem'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='stem'),
            post_act_block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv1', conv_type='spconv'),
        )

        # [944, 944, 21] -> [472, 472, 11]
        self.conv2 = spconv.SparseSequential(
            SEDLayer(32, down_kernel_size, down_stride, num_SBB, norm_fn=norm_fn, indice_key='sedlayer2'),
            post_act_block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
        )

        #  [472, 472, 11] -> [236, 236, 11]
        self.conv3 = spconv.SparseSequential(
            SEDLayer(64, down_kernel_size, down_stride, num_SBB, norm_fn=norm_fn, indice_key='sedlayer3'),
            post_act_block(64, dim, 3, norm_fn=norm_fn, stride=(1, 2, 2), padding=1, indice_key='spconv3', conv_type='spconv'),
        )

        self.layers = nn.ModuleList()
        for idx in range(num_layers):
            conv = SEDLayer(dim, down_kernel_size, down_stride, num_SBB, norm_fn=norm_fn, indice_key=f'sedlayer{idx+4}')
            self.layers.append(conv)

        # [236, 236, 11] -> [236, 236, 5] --> [236, 236, 2]
        self.conv_out = spconv.SparseSequential(
            spconv.SparseConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=0, bias=False, indice_key='spconv4'),
            norm_fn(dim),
            nn.ReLU(),
            spconv.SparseConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=0, bias=False, indice_key='spconv5'),
            norm_fn(dim),
            nn.ReLU(),
        )

        self.num_point_features = dim
        self.init_weights()

    def init_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, (spconv.SubMConv3d, spconv.SparseConv3d)):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, voxel_features, voxel_coords, batch_size):
        x = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        for conv in self.layers:
            x = conv(x)
        out = self.conv_out(x).dense()

        N, C, D, H, W = out.shape
        out = out.view(N, C * D, H, W)
        return out
