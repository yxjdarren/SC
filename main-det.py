import os
import sys
from mim.cli import cli

os.environ["PYTHONPATH"] = '.:' + os.environ.get("PYTHONPATH", "")

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

desired_directory = '/home/'
os.chdir(desired_directory)

sys.path.insert(0, desired_directory)

sys.argv = [
    'mim',
    'test',
    'mmdet',

    '/home/configs/det/linear_probe/faster_rcnn/central_r50_fpn_Up-G-D_pretrain_voc0712_10p.py',

    '--checkpoint',"/home/work_dirs/pth-det/best.pth",

    '--eval', 'mAP'
]

cli()
