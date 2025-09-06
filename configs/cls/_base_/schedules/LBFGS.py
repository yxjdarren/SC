# optimizer
optimizer = dict(
    type='LBFGS',
    lr=0.01,
    max_iter=1000, 
    line_search_fn='strong_wolfe'
)

optimizer_config = dict(grad_clip=None)

# learning policy
#lr_config = dict(policy='CosineAnnealing', min_lr=0.0, warmup='linear', warmup_iters=250,warmup_ratio=0.001)
lr_config = dict(policy='fixed')
runner = dict(type='IterBasedRunner', max_iters=1000)
