import math

import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmdet.models.backbones.resnet import ResNet
from mmdet.models.builder import BACKBONES

from .metanet import (Block, FusedMBConv3x3, MBConv3x3, MetaNet, SABlock, to3d,
                      to4d)
from .trans_block import crossconvhrnetlayer

#region evolutionary
import os
from mmcv.runner import Hook, HOOKS, CheckpointHook
#endregion

BN = torch.nn.BatchNorm2d
Conv2d = torch.nn.Conv2d

#region evolutionary

@HOOKS.register_module()
class IterationLoggerHook(Hook):
    def before_train_epoch(self, runner):
        # check runner.model or runner.model.module
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        model.backbone.current_epoch = runner.epoch  # update current_epoch
        print(f"epoch: {runner.epoch}")

class CosineDecay(object):
    def __init__(self,
                max_value,
                min_value,
                num_loops):
        self._max_value = max_value
        self._min_value = min_value
        self._num_loops = num_loops

    def get_value(self, i):
        if i < 0:
            i = 0
        if i >= self._num_loops:
        # if i >= 12:
            i = 8
        # value = (math.tan(i * math.pi / self._num_loops))
        value = (math.sin(i * math.pi / self._num_loops))
        value = abs(value) * (self._max_value - self._min_value) + self._min_value
        return value

    # def get_value(self, i):
    #     if i < 0:
    #         i = 0
    #     if i >= self._num_loops:
    #         i = self._num_loops
    #     value = (math.cos(i * math.pi / self._num_loops) + 1.0) * 0.5
    #     value = value * (self._max_value - self._min_value) + self._min_value
    #     return value
#endregion

@BACKBONES.register_module()
class Central_Model(BaseModule):
    def __init__(self,
                 task_name_to_backbone,
                 backbone_name,
                 task_names=('gv_patch', 'gv_global'),
                 main_task_name='gv_patch',
                 frozen_stages=4,
                 # frozen_stages=0,
                 freeze_at=0,
                 trans_type='scalablelayer',
                 trans_layers=['layer1', 'layer2', 'layer3'],
                 channels=[256, 512, 1024],
                 init_cfg=[],
                 **kwargs):

        super(Central_Model, self).__init__(init_cfg)

        self.frozen_stages = frozen_stages
        self.local_task = main_task_name
        self.model = nn.ModuleDict()
        self.model['backbone'] = nn.ModuleDict()
        self.name = backbone_name

        #region evolutionary
        self.current_epoch = 0
        self.printed_iters = set()
        self.ConfidNet_main = nn.Sequential(
            nn.Linear(2048, 2048*2),
            nn.Linear(2048*2, 2048),
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )

        self.ConfidNet_aux = nn.Sequential(
            nn.Linear(2048, 2048*2),
            nn.Linear(2048*2, 2048),
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )
        #endregion

        for task_name in task_names:
            if 'resnet' in self.name:
                self.model['backbone'][task_name] = ResNet(
                    **task_name_to_backbone[task_name])
            else:
                self.model['backbone'][task_name] = MetaNet(
                    **task_name_to_backbone[task_name])

        self.main_task_name = main_task_name
        self.task_names = task_names#('gv_patch', 'gv_global')
        self.tasks_set = set(task_names)#{'gv_global', 'gv_patch'}
        self.use_cbn = False
        self.trans_use_cross = 'cross' in trans_type#True
        self.freeze_at = freeze_at
        self.trans_layers = trans_layers#['layer1', 'layer2', 'layer3']
        self.channels = channels#[64, 128, 192]

        trans = nn.ModuleDict()#ModuleDict()
        self.model['trans'] = trans
        trans_entry = crossconvhrnetlayer

        self.backbone_stage = {}
        for task_name in self.tasks_set:
            trans[task_name] = nn.ModuleDict()
            self.backbone_stage[task_name] = nn.ModuleDict()
            # pre-stage
            if 'resnet' in self.name:
                self.backbone_stage[task_name]['pre_layer'] = nn.Sequential(
                    self.backbone[task_name].conv1,
                    self.backbone[task_name].bn1,
                    self.backbone[task_name].relu,
                    self.backbone[task_name].maxpool)
            if 'resnet' in self.name:
                pool_layer = self.backbone_stage[task_name]['pre_layer'][-1]
                self.backbone_stage[task_name]['layer1'] = nn.Sequential(
                    pool_layer, self.backbone[task_name].layer1)

                self.backbone_stage[task_name][
                    'pre_layer'] = self.backbone_stage[task_name][
                        'pre_layer'][:-1]
                self.backbone_stage[task_name]['layer2'] = self.backbone[
                    task_name].layer2
                self.backbone_stage[task_name]['layer3'] = self.backbone[
                    task_name].layer3
                self.backbone_stage[task_name]['layer4'] = self.backbone[
                    task_name].layer4
                self.backbone_stage[task_name][
                    'pre_fc'] = nn.AdaptiveAvgPool2d((1, 1))
            elif 'MTB' in self.name:
                self.backbone_stage[task_name] = self.backbone[task_name]

            for auxiliary_task in self.tasks_set:
                if auxiliary_task == task_name:
                    continue
                trans[task_name][auxiliary_task] = nn.ModuleDict()

                for trans_layer in range(len(trans_layers)):
                    if 'hrnet' in trans_type:
                        trans[task_name][auxiliary_task][
                            trans_layers[trans_layer]] = trans_entry(
                                **{'channel': channels[trans_layer]},
                                name=trans_layers[trans_layer],
                                **kwargs)
                    else:
                        trans[task_name][auxiliary_task][
                            trans_layers[trans_layer]] = trans_entry(
                                **{'channel': channels[trans_layer]}, )

        self._freeze_stages()

    @property
    def backbone(self):
        return self.model['backbone']

    @property
    def trans(self):
        return self.model['trans']

    def forward_features(self, inputs):
        inp_feature = {}
        if 'resnet' in self.name:
            last_stage = [
                'input', 'pre_layer', 'layer1', 'layer2', 'layer3', 'layer4'
            ]
            cur_stage = [
                'pre_layer', 'layer1', 'layer2', 'layer3', 'layer4', 'pre_fc'
            ]
        elif 'MTB' in self.name:
            last_stage = ['input', 'layer1', 'layer2', 'layer3', 'layer4']
            cur_stage = ['layer1', 'layer2', 'layer3', 'layer4', 'pre_fc']
        for task_name in self.tasks_set:
            inp_feature[task_name] = {}#{'gv_global': {}, 'gv_patch': {}}

        for task_name in self.tasks_set:
            inp_feature[task_name]['input'] = inputs#{'gv_global': {input}, 'gv_patch': {input}} input = tensor of(2,3,928,608)

        task_names = self.tasks_set.copy()#{'gv_global', 'gv_patch'}

        if not self.trans_use_cross: #self.trans_use_cross = true
            task_names.remove(self.local_task)
            task_names.insert(0, self.local_task)
            for idx in range(len(cur_stage)):
                stage = cur_stage[idx]
                for task_name in task_names:
                    if 'resnet' in self.name:
                        x = self.backbone_stage[task_name][stage](
                            inp_feature[task_name][last_stage[idx]])
                    if stage == 'pre_fc':
                        if 'resnet' in self.name:
                            x = torch.flatten(x, 1)
                    inp_feature[task_name][stage] = x
                if stage in self.trans_layers:
                    for task_name in self.tasks_set:
                        if task_name != self.local_task:
                            x = self.trans[self.local_task][task_name][stage](
                                inp_feature[task_name][stage].detach(),
                                inp_feature[self.local_task][stage])
                            inp_feature[self.local_task][stage] = inp_feature[
                                self.local_task][stage] + x
        else:# start trans
            if 'MTB' in self.name:
                for task_name in task_names:
                    if task_name != self.local_task:
                        x = self.backbone_stage[task_name].stem(
                            inp_feature[task_name]['input'])
                        layer = 0
                        b, c, h, w = x.shape
                        for i, blk in enumerate(
                                self.backbone_stage[task_name].blocks):
                            if isinstance(blk, (SABlock, Block)):
                                x = to3d(x)
                            else:
                                assert isinstance(blk,
                                                  (MBConv3x3, FusedMBConv3x3))
                                x = to4d(x, h, w)
                            x = blk(x, h, w)
                            h = math.ceil(
                                h / self.backbone_stage[task_name].scale[i])
                            w = math.ceil(
                                w / self.backbone_stage[task_name].scale[i])
                            if layer < len(self.backbone_stage[
                                task_name
                            ].out_blocks) and i == \
                                    self.backbone_stage[
                                        task_name
                                    ].out_blocks[layer]:
                                inp_feature[task_name][
                                    f'layer{layer + 1}'] = to4d(x, h, w)
                                layer += 1
                                if layer == 3:
                                    break
            else:
                for idx in range(len(cur_stage)):#cur_stage=['pre_layer', 'layer1', 'layer2', 'layer3', 'layer4', 'pre_fc']
                    stage = cur_stage[idx]# start from 'pre_layer'
                    # forward all aux backbone first
                    for task_name in task_names:
                        if task_name != self.local_task:
                            if 'resnet' in self.name:
                                x = self.backbone_stage[task_name][stage](
                                    inp_feature[task_name][last_stage[idx]])

                            if stage == 'pre_fc':
                                if 'resnet' in self.name:
                                    x = torch.flatten(x, 1)
                            inp_feature[task_name][stage] = x
            if 'MTB' in self.name:
                x = self.backbone_stage[self.local_task].stem(
                    inp_feature[self.local_task]['input'])
                layer = 0
                b, c, h, w = x.shape
                for i, blk in enumerate(
                        self.backbone_stage[self.local_task].blocks):
                    if isinstance(blk, (SABlock, Block)):
                        x = to3d(x)
                    else:
                        assert isinstance(blk, (MBConv3x3, FusedMBConv3x3))
                        x = to4d(x, h, w)
                    x = blk(x, h, w)
                    h = math.ceil(
                        h / self.backbone_stage[self.local_task].scale[i])
                    w = math.ceil(
                        w / self.backbone_stage[self.local_task].scale[i])
                    if layer < len(
                        self.backbone_stage[self.local_task].out_blocks
                    ) and i == \
                            self.backbone_stage[
                                self.local_task
                            ].out_blocks[layer]:
                        stage = f'layer{layer + 1}'
                        inp_feature[self.local_task][stage] = to4d(x, h, w)
                        if stage in self.trans_layers:
                            for task_name in self.tasks_set:
                                if task_name != self.local_task:
                                    x = self.trans[
                                        self.local_task][task_name][stage](
                                            inp_feature[task_name],
                                            inp_feature[
                                                self.local_task][stage],
                                            detach=True)
                                    inp_feature[
                                        self.local_task][stage] = inp_feature[
                                            self.local_task][stage] + x
                            x = inp_feature[self.local_task][stage]
                        layer += 1

                x = to4d(x, h, w)
                x = self.backbone_stage[self.local_task].head(x)
                x = self.backbone_stage[self.local_task].avgpool(x)
                x = torch.flatten(x, 1)
                x = self.backbone_stage[self.local_task].final_drop(x)
                inp_feature[self.local_task]['pre_fc'] = x
            else:
                for idx in range(len(cur_stage)):
                    stage = cur_stage[idx]
                    # forward main backbone second
                    if 'resnet' in self.name:
                        x = self.backbone_stage[self.local_task][stage](
                            inp_feature[self.local_task][last_stage[idx]])
                    if stage == 'pre_fc':
                        if 'resnet' in self.name:
                            x = torch.flatten(x, 1)
                    inp_feature[self.local_task][stage] = x

                    if stage in self.trans_layers:
                        for task_name in self.tasks_set:
                            if task_name != self.local_task:
                                a = self.local_task
                                x = self.trans[
                                    self.local_task][task_name][stage](
                                    #self.trans[
                                    #self.local_task][task_name][stage](
                                        inp_feature[task_name],
                                        inp_feature[self.local_task][stage],
                                        detach=True)

                                # inp_feature[
                                #     self.local_task][stage] = inp_feature[
                                #                                   self.local_task][stage] + x

                                # region evolutionary
                                # gradient_decay = CosineDecay(max_value=1, min_value=0,
                                #                              num_loops=8)
                                # decay_value = gradient_decay.get_value(self.current_epoch)
                                decay_value=0.59
                                # print(f"epoch: {self.current_epoch}, lambda_aux: {decay_value}")
                                if self.current_epoch not in self.printed_iters:
                                    print(f"epoch: {self.current_epoch}, lambda_aux: {decay_value}")
                                    self.printed_iters.add(self.current_epoch)  # 将当前 epoch 添加到集合中

                                inp_feature[
                                    self.local_task][stage] = inp_feature[
                                                                  self.local_task][stage] + (decay_value) * x
                                # endregion

                            # region evolutionary
                            if stage == 'pre_fc':
                                for task_name in self.tasks_set:
                                    if task_name != self.local_task:
                                        # region evolutionary
                                        feature_main_cp = inp_feature[self.local_task][stage].clone().detach()
                                        feature_aux_cp = inp_feature[task_name][stage].clone().detach()
                                        main_tcp = self.ConfidNet_main(feature_main_cp)  # Mono-Confidence
                                        aux_tcp = self.ConfidNet_aux(
                                            feature_aux_cp)

                                        main_holo = torch.log(aux_tcp) / (
                                                    torch.log(main_tcp * aux_tcp) + 1e-8)  # Holo-Confidence
                                        aux_holo = torch.log(main_tcp) / (
                                                    torch.log(main_tcp * aux_tcp) + 1e-8)  # Holo-Confidence
                                        cb_main = main_tcp.detach() + main_holo.detach()  # co-belief
                                        cb_aux = aux_tcp.detach() + aux_holo.detach()  # co-belief
                                        w_all = torch.stack((cb_main, cb_aux), 1)
                                        softmax = nn.Softmax(1)
                                        w_all = softmax(w_all)
                                        w_main = w_all[:, 0]
                                        w_aux = w_all[:, 1]
                                        main_aux_out = w_main.detach() * feature_main_cp + w_aux.detach() * feature_aux_cp

                                        # main_aux_out = feature_main_cp + 0 * feature_aux_cp

                                        inp_feature[
                                            self.local_task][stage] = main_aux_out
                                        # endregion
                            # endregion

        out = {
            # 'pre_layer': inp_feature[self.local_task]['pre_layer'],
            'layer1': inp_feature[self.local_task]['layer1'],
            'layer2': inp_feature[self.local_task]['layer2'],
            'layer3': inp_feature[self.local_task]['layer3'],
            'layer4': inp_feature[self.local_task]['layer4'],
            'pre_fc': inp_feature[self.local_task]['pre_fc'],
        }

        return out

    def forward(self, x):
        with torch.no_grad():
            outs = self.forward_features(x)

        return tuple(
            [outs['layer1'], outs['layer2'], outs['layer3'], outs['layer4']])

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for k, (name, m) in enumerate(self.named_modules()):
                if k == 0:
                    continue
                else:
                    m.eval()
                    for param in m.parameters():
                        param.requires_grad = False
        else:
            pass

    def train(self, mode=True):
        super(Central_Model, self).train(mode)
        self._freeze_stages()
        if mode:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
