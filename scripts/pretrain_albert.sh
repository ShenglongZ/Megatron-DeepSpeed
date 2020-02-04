#!/bin/bash

RANK=0
WORLD_SIZE=1

python pretrain_albert.py \
       --num-layers 12 \
       --hidden-size 768 \
       --num-attention-heads 12 \
       --batch-size 4 \
       --seq-length 512 \
       --max-preds-per-seq 80 \
       --max-position-embeddings 512 \
       --train-iters 10000 \
       --save checkpoints/albert_117m \
       --load checkpoints/albert_117m \
       --resume-dataloader \
       --data-path data/megatron/bc_rn_owt_sto_wiki_dedup_shuf_cleaned_0.7_mmap \
       --vocab data/megatron/vocab.txt \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --lr-decay-iters 990000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --fp16 \
       --fp32-layernorm \
       --fp32-embedding \
       --skip-mmap-warmup \
       --num-workers 0