import torch
from mmcv import Config
from mmcls.models import build_classifier
from mmcls.datasets import build_dataset, build_dataloader
from mmcv.runner import load_checkpoint
from mmcls.apis import single_gpu_test
from mmcv.parallel import MMDataParallel
from mmcls.models.losses import accuracy

import pickle


cfg = Config.fromfile('/home/configs/cls/linear_probe/test_r50_Up-G-C_pretrain_cifar100_10p.py')


datasets = [build_dataset(cfg.data.test)]
data_loader = build_dataloader(
    datasets[0],
    samples_per_gpu=cfg.data.samples_per_gpu,
    workers_per_gpu=cfg.data.workers_per_gpu,
    shuffle=False
)


model = build_classifier(cfg.model)
model.CLASSES = datasets[0].CLASSES


checkpoint = load_checkpoint(
    model,
    '/home/work_dirs/r50_Up-G-C_pretrain_cifar100_10p/fish_20241109/best_accuracy_top-1_iter_6074.pth',
    map_location='cpu',
    strict=False
)


model = MMDataParallel(model.cuda(), device_ids=[0])


print("Starting inference...")
evaluation = dict(interval=1, start=1, metric='accuracy')

# outputs = single_gpu_test(model, data_loader)
#
#
# output_file = '/home/work_dirs/G(D)-TEST-C(cifar100_10p)(pth_best_mAP_epoch_16_G-D-voc07+12)/result.pkl'
# with open(output_file, 'wb') as f:
#     pickle.dump(outputs, f)
#
# print(f"Inference completed. Results saved to {output_file}.")
#
#
# print("Calculating accuracy...")
#
# true_labels = [dataset['gt_labels'] for dataset in datasets[0]]
# top1_acc = accuracy(torch.tensor(outputs), torch.tensor(true_labels), topk=(1,))
#
# print(f"Top-1 Accuracy: {top1_acc.item() * 100:.2f}%")
