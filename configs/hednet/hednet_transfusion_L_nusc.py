_base_=['./simple_transfusion_L_nusc.py']

model = dict(
    pts_middle_encoder=dict(
        _delete_=True,
        type='HEDNet',
        in_channels=64,
        sparse_shape=[41, 1440, 1440],
        model_cfg=dict(
            FEATURE_DIM=64,
            NUM_LAYERS=2,
            NUM_SBB=[2, 1, 1],
            DOWN_STRIDE=[1, 2, 2],
            DOWN_KERNEL_SIZE=[3, 3, 3],
        )
    ),
    pts_backbone=dict(
        _delete_=True,
        type='CascadeDEDBackbone',
        in_channels=128,
        model_cfg=dict(
            FEATURE_DIM=128,
            NUM_LAYERS=4,
            NUM_SBB=[2, 1, 1],
            DOWN_STRIDES=[1, 2, 2],
        )
    ),
    pts_neck=None,
    pts_bbox_head=dict(in_channels=128)
)
