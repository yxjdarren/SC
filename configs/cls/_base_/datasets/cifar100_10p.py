# dataset settings
CLASSES = [
    'apple', 'aquarium fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak tree', 'orange', 'orchid', 'otter', 'palm tree', 'pear', 'pickup truck', 'pine tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet pepper', 'table', 'tank', 'telephone', 'television', 'tiger',
    'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow tree', 'wolf', 'woman', 'worm'
]


dataset_type = 'CIFAR_100'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),  # **
    dict(type='Resize', size=(224, -1), backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(224, -1), backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
dataset_num_classes = 100
data = dict(
    samples_per_gpu=64,  # batchsize
    workers_per_gpu=2,
    train=dict(type=dataset_type,
               data_prefix='/home/data_use/cifar-100/train_pro',
               ann_file= '/home/data_use/cifar-100//train_10p.list',
               pipeline=train_pipeline),
    val=dict(type=dataset_type,
             data_prefix='/home/data_use/cifar-100/test_pro',
             ann_file='/home/data_use/cifar-100//test.list',
             pipeline=test_pipeline),
    test=dict(type=dataset_type,
              classes = CLASSES,
              data_prefix='/home/data_use/cifar-100/test_pro',
              ann_file='/home/data_use/cifar-100//test.list',
              pipeline=test_pipeline))

#region evolutionary
# evaluation = dict(interval=100, metric='accuracy')
# evaluation = dict(interval=100, metric='per_class_acc')
evaluation = dict(interval=100, start=9000, metric='accuracy', metric_options={'topk': (1, )}, save_best='auto')
#endregion

