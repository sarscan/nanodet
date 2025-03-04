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

from collections import OrderedDict
from typing import Any, Dict

import pytorch_lightning as pl
import torch

from .rank_filter import rank_filter


def load_model_weight(model, checkpoint, logger):
    state_dict = checkpoint["state_dict"].copy()

    # with open('state_dict_keys.txt', 'w') as f:
    #     for key in state_dict.keys():
    #         f.write(f"{key}\n")

    for k in checkpoint["state_dict"]:
        # convert average model weights
        # gsc 我打印了ckpt里所有的key，前一半是model.开头，后一半是avg_model开头，所以先把avg_去掉再统一去掉model.
        # 那c++推onnx是怎么做的呢？导出onnx的时候也调用了本函数啊，然后onnx再转ncnn的格式，没问题
        if k.startswith("avg_model."):
            v = state_dict.pop(k)
            state_dict[k[4:]] = v
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    if list(state_dict.keys())[0].startswith("model."):
        state_dict = {k[6:]: v for k, v in state_dict.items()}

    model_state_dict = (
        model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    )

    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                logger.log(
                    "Skip loading parameter {}, required shape{}, "
                    "loaded shape{}.".format(
                        k, model_state_dict[k].shape, state_dict[k].shape
                    )
                )
                state_dict[k] = model_state_dict[k]	 # bad case1 就是说虽然匹配到key，但是参数的shape不一样，就还是以model已有的参数为准
        else:
            logger.log("Drop parameter {}.".format(k))  # bad case2 ckpt里多了参数
    for k in model_state_dict:
        if not (k in state_dict):
            logger.log("No param {}.".format(k))
            state_dict[k] = model_state_dict[k]	 # bad case3 同理，ckpt里少了参数，也是以model自己的为准
    model.load_state_dict(state_dict, strict=False)	 # strict=false其实已经处理上述全部三个cases，上面只是为了加打印


@rank_filter
def save_model(model, path, epoch, iter, optimizer=None):
    model_state_dict = (
        model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    )
    data = {"epoch": epoch, "state_dict": model_state_dict, "iter": iter}
    if optimizer is not None:
        data["optimizer"] = optimizer.state_dict()

    torch.save(data, path)


def convert_old_model(old_model_dict):
    if "pytorch-lightning_version" in old_model_dict:
        raise ValueError("This model is not old format. No need to convert!")
    version = pl.__version__
    epoch = old_model_dict["epoch"]
    global_step = old_model_dict["iter"]
    state_dict = old_model_dict["state_dict"]
    new_state_dict = OrderedDict()
    for name, value in state_dict.items():
        new_state_dict["model." + name] = value

    new_checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "pytorch-lightning_version": version,
        "state_dict": new_state_dict,
        "lr_schedulers": [],
    }

    if "optimizer" in old_model_dict:
        optimizer_states = [old_model_dict["optimizer"]]
        new_checkpoint["optimizer_states"] = optimizer_states

    return new_checkpoint


def convert_avg_params(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Converts average state dict to the format that can be loaded to a model.
    Args:
        checkpoint: model.
    Returns:
        Converted average state dict.
    """
    state_dict = checkpoint["state_dict"]
    avg_weights = {}
    for k, v in state_dict.items():
        if "avg_model" in k:
            avg_weights[k[10:]] = v
    return avg_weights
