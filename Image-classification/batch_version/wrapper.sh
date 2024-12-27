#!/bin/bash
dir=/ibex/user/shaima0d/KSL_Trainings/hpc-saudi-2024/tiny_imagenet_ds
    python -m torch.distributed.launch --use_env --nproc_per_node=${SLURM_GPUS_PER_NODE} \
    --nnodes=${SLURM_NNODES} --node_rank=${SLURM_NODEID} \
    --master_addr=${master_ip} --master_port=${MASTER_PORT} \
    ${dir}/scripts/train_resnet50_ds.py  --epochs ${EPOCHS} --num-workers=${SLURM_CPUS_PER_TASK} \
    --deepspeed --deepspeed_config ${dir}/scripts/ds_config.json \
    --log-interval 50