_base_=['./simple_transfusion_nusc_voxel_L.py']

point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
voxel_size = [0.075, 0.075, 0.2]

model = dict(
    type='SimpleTransFusionDV',
    pts_voxel_layer=dict(
        _delete_=True,
        max_num_points=-1,
        voxel_size=voxel_size,
        max_voxels=(-1, -1),
        point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(
        _delete_=True,
        type='DynamicVFE',
        in_channels=5,
        feat_channels=[64, 64],
        with_distance=False,
        with_cluster_center=True,
        with_voxel_center=True,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range),
    pts_middle_encoder=dict(in_channels=64),
)
