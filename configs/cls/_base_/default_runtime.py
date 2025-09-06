# 检查点配置，去掉了重复的 evaluation
checkpoint_config = dict(interval=5000, max_keep_ckpts=1)

# 日志配置
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

# 分布式参数
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 10000), ('val', 5000)]

#region evolutionary
# 自定义钩子，包括迭代记录和 EvalHook
# custom_hooks = [
#     dict(type='IterationLoggerHook')
# ]
# #endregion