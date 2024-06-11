export CUDA_VISIBLE_DEVICES=0
model_name=PatchTST
pretrain_seq_len=96
mask_rate=0.25
sampling_range=6
lineage_tokens=2
representation_using=avg

python -u run.py \
    --task_name timesiam \
    --is_training 0 \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --model_id traffic \
    --model $model_name \
    --data Traffic \
    --features M \
    --seq_len $pretrain_seq_len \
    --e_layers 3 \
    --d_layers 1 \
    --d_model 128 \
    --d_ff 256 \
    --n_heads 8 \
    --patch_len 12 \
    --stride 12 \
    --mask_rate $mask_rate \
    --sampling_range $sampling_range \
    --lineage_tokens $lineage_tokens \
    --train_epochs 50 \
    --batch_size 16

checkpoint=./outputs/pretrain_checkpoints/Traffic/ckpt_best.pth

for pred_len in 96 192 336 720; do
    python -u run.py \
        --task_name fine_tune \
        --is_training 1 \
        --root_path ./dataset/traffic/ \
        --data_path traffic.csv \
        --model_id traffic \
        --model $model_name \
        --data Traffic \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 3 \
        --d_layers 1 \
        --d_model 128 \
        --d_ff 256 \
        --n_heads 8 \
        --patch_len 12 \
        --stride 12 \
        --factor 3 \
        --lineage_tokens $lineage_tokens \
        --representation_using $representation_using \
        --load_checkpoints $checkpoint \
        --learning_rate 0.001 \
        --batch_size 4 \
        --head_dropout 0
done
