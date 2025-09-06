_base_ = ['./faster_rcnn_r50_fpn.py']

model = dict(
    type='FasterRCNN',
    backbone=dict(
        frozen_stages=4,
        # frozen_stages=0,
        init_cfg=dict(type='Pretrained',
                                # checkpoint='/home/checkpoint/INTERN-r50-Up-E-D-7ffb70c57.pth.tar')),
                                checkpoint='/home/checkpoint/INTERN-r50-Up-E-C-65a418071.pth.tar')),
rpn_head=dict(
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(bbox_head=dict(
        num_classes=1,
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    train_cfg=dict(rpn=dict(allowed_border=0),
                   rpn_proposal=dict(nms_across_levels=False,
                                     nms_pre=2000,
                                     nms_post=2000,
                                     max_num=2000,
                                     nms_thr=0.7,
                                     min_bbox_size=0)),
    test_cfg=dict(rpn=dict(nms_across_levels=False,
                           nms_pre=1000,
                           nms_post=1000,
                           max_num=1000,
                           nms_thr=0.7,
                           min_bbox_size=0)))
