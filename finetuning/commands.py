# %%
""" Download model and test"""
# enter the src folder of the open_clip repository
cd open_clip/src

# specify which GPUs you want to use.
export CUDA_VISIBLE_DEVICES=1,2,3,4,5

# set the training args

nohup torchrun  --standalone --rdzv_backend=c10d --nproc_per_node 5 -m open_clip_train.main -- \
    --batch-size 64 \
    --precision amp \
    --workers 4 \
    --report-to wandb \
    --save-frequency 10 \
    --save-most-recent \
    --logs /shared/scratch/0/home/v_neelesh_bisht/projects/open_clip/finetuning/logs \
    --dataset-type csv \
    --csv-separator="," \
    --train-data /shared/scratch/0/home/v_neelesh_bisht/projects/open_clip/dataset/cub_finetune_100cls_train.csv \
    --val-data /shared/scratch/0/home/v_neelesh_bisht/projects/open_clip/dataset/cub_finetune_100cls_val.csv \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --warmup 200 \
    --lr=1e-5 \
    --wd=0.05 \
    --epochs=100 \
    --model ViT-B-32 \
    --pretrained "laion2b_s34b_b79k" \
    --name cub_100cls_vitb32 \
    --resume latest \
    --seed 42 \
    > train_resume_log_2.out 2>&1 &


wandb api key:
    208a19b5d5e788a2ca3f0f24b3c2b0f2fc32544e




export WANDB_ENTITY=neeleshbisht99-carnegie-mellon-university
export WANDB_PROJECT=openclip_finetune

export WANDB_NAME="cub_3cls_vitb32"
export WANDB_DIR="/shared/scratch/0/home/v_neelesh_bisht/projects/open_clip/finetuning/logs"
export WANDB_TAGS="vitb32,cub,3cls"