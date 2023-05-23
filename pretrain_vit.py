# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Pretrain VIT"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron import get_args, get_timers, mpu, print_rank_0
from megatron.data.vit_dataset import build_train_valid_datasets
from megatron.model.vit_model import VitModel
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group
from megatron.mpu.initialize import get_tensor_model_parallel_group, get_tensor_model_parallel_src_rank

from deepspeed.accelerator.real_accelerator import get_accelerator
def model_provider():
    """Build the model."""

    print_rank_0("building VIT model ...")
    args = get_args()

    model = VitModel(num_classes=args.num_classes)
    return model

def get_batch(data_iterator):
    """Build the batch."""
    data = next(data_iterator)

    # only data parallelism; no need for broadcast
    images = data[0].to(get_accelerator().device_name())
    labels = data[1].to(get_accelerator().device_name())

    return images, labels

def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()

    # Get the batch.

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
        images = data[0].to(get_accelerator().device_name())
        labels = data[1].to(get_accelerator().device_name())
        # print(type(data))
        # print(len(data))
        # print(data[0].size())
        # print(data[1].size())
    else:
        data = None
        images = torch.empty([16, 3, 224, 224]).to(get_accelerator().device_name())
        labels = torch.empty([16]).to(get_accelerator().device_name())
    torch.distributed.broadcast(images, get_tensor_model_parallel_src_rank(),
                                group=get_tensor_model_parallel_group())
    torch.distributed.broadcast(labels, get_tensor_model_parallel_src_rank(),
                                group=get_tensor_model_parallel_group())
    torch.distributed.barrier()
    torch.cuda.synchronize()
    # data_b = mpu.broadcast_data(keys, data, datatype)
    train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.cuda()
    # timers("batch-generator").start()
    # (
    #     images,
    #     labels,
    # ) = get_batch(data_iterator)
    # timers("batch-generator").stop()

    # Forward model. lm_labels
    logits = model(images).contiguous().float()

    labels = labels.type(torch.LongTensor).cuda()
    loss = train_loss_fn(logits, labels)

    outputs = torch.argmax(logits, -1)
    correct = (outputs == labels).float()
    accuracy = torch.mean(correct)

    averaged_loss = average_losses_across_data_parallel_group([loss, accuracy])

    return loss, {"loss": averaged_loss[0], "accuracy": averaged_loss[1]}


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0(
        "> building train, validation, and test datasets " "for VIT ..."
    )
    train_ds, _ = build_train_valid_datasets(data_path=args.data_path)
    print_rank_0("> finished creating VIT datasets ...")

    return train_ds, None, None


if __name__ == "__main__":

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        forward_step,
        args_defaults={'dataloader_type': 'cyclic'}
    )
