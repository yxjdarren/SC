import os
import sys
from mim.cli import cli

os.environ["PYTHONPATH"] = '.:' + os.environ.get("PYTHONPATH", "")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

desired_directory = '/home/'
os.chdir(desired_directory)

sys.path.insert(0, desired_directory)

sys.argv = [
    'mim',
    'test',
    'mmcls',

    '/home/configs/cls/linear_probe/r50_Up-G-C_pretrain_cifar100_10p.py',

    '--checkpoint', "/home/work_dirs/pth-cls/best.pth",

    '--metrics', 'accuracy'
]

cli()
