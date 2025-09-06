_base_ = [
    '../../_base_/models/faster_rcnn_central_r50_fpn_intern.py',
    '../../_base_/datasets/voc0712_10p.py',
    '../../_base_/schedules/schedule.py', '../../_base_/default_runtime.py'
]

model = dict(backbone=dict(init_cfg=dict(
    type='Pretrained',
    checkpoint="/home/checkpoint/INTERN-r50-Up-G-dbn-a4040c9c4.pth.tar",
    # checkpoint="/home/checkpoint/INTERN-r50-Up-G-cbn-128cdf619.pth.tar",
)),
             roi_head=dict(bbox_head=dict(num_classes=20)))

custom_imports = dict(imports=[
    'gvbenchmark.det.models.backbones.central_model',
],
                      allow_failed_imports=False)
