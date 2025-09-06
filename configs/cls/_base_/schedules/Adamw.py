# optimizer 0.01_wd_0.001_0.9
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.1)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.000001,
                 warmup='linear',
                 warmup_iters=2500,
                 warmup_ratio=0.001
                 )

runner = dict(type='IterBasedRunner', max_iters=100)
