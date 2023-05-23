#! /bin/bash

# Runs the "345M" parameter model

# Change for multinode config
DATA_PATH=/mnt/petrelfs/share/images
T=`date +%m%d%H%M`
 
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29600}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
MASTER_PORT=29600

WORLD_SIZE=8
TENSOR_MP_SIZE=2

python -m torch.distributed.launch \
       --nnodes=$NNODES \
       --node_rank=$NODE_RANK \
       --master_addr=$MASTER_ADDR \
       --nproc_per_node=8 \
       --master_port=$PORT \
       pretrain_vit.py \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 16 \
       --global-batch-size 128 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 500000 \
       --lr-decay-iters 320000 \
       --data-path $DATA_PATH \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --checkpoint-activations \
       --log-interval 100 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --no-pipeline-parallel \
       --tensor-model-parallel-size $TENSOR_MP_SIZE \
       --deepspeed \
       --deepspeed_config /mnt/petrelfs/zhangshenglong/Megatron-DeepSpeed/deepspeed_config/zero1_fp16.json \
       --fp16 2>&1 | tee  test_$T.log
