# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

import torch
import torch.nn as nn

from ..backbone import build_backbone
from ..fpn import build_fpn
from ..head import build_head


class OneStageDetector(nn.Module):
    def __init__(
        self,
        backbone_cfg,
        fpn_cfg=None,
        head_cfg=None,
    ):
        super(OneStageDetector, self).__init__()
        self.backbone = build_backbone(backbone_cfg)
        if fpn_cfg is not None:
            self.fpn = build_fpn(fpn_cfg)
        if head_cfg is not None:
            self.head = build_head(head_cfg)
        self.epoch = 0

    def forward(self, x):
        x = self.backbone(x)    # 这几个输入、输出是怎么级联的，backbone 输出的是一个turple啊，只要FPN里forward函数解析长度为3的turple即可
        if hasattr(self, "fpn"):
            x = self.fpn(x)     # gsc *fpn也全是conv norm relu这些东西，最后竟然能检测出obj，也是神奇，其实就是各种filter的组合，也合理
        if hasattr(self, "head"):
            x = self.head(x)
        return x

    # 这个接口就是demo.py里调用的
    def inference(self, meta):
        with torch.no_grad():
            is_cuda_available = torch.cuda.is_available()
            if is_cuda_available:
                torch.cuda.synchronize()

            time1 = time.time()
            preds = self(meta["img"])  # Module重载了运算符()，会调用上面的forward函数

            if is_cuda_available:
                torch.cuda.synchronize()

            time2 = time.time()
            print("forward time: {:.3f}s".format((time2 - time1)), end=" | ")
            results = self.head.post_process(preds, meta)

            if is_cuda_available:
                torch.cuda.synchronize()

            print("decode time: {:.3f}s".format((time.time() - time2)), end=" | ")
        return results

    def forward_train(self, gt_meta):
        preds = self(gt_meta["img"])
        loss, loss_states = self.head.loss(preds, gt_meta)

        return preds, loss, loss_states

    def set_epoch(self, epoch):
        self.epoch = epoch
