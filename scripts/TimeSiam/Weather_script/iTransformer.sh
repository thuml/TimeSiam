export CUDA_VISIBLE_DEVICES=0
model_name=PatchTST
train_epochs=50
mask_rate=0.25
pretrain_seq_len=96
sampling_range=6
lineage_tokens=2
representation_using=concat

python -u run.py \
    --task_name timesiam \
    --is_training 0 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id weather \
    --model $model_name \
    --data Weather \
    --features M \
    --seq_len $pretrain_seq_len \
    --e_layers 2 \
    --d_layers 1 \
    --n_heads 4 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --mask_rate $mask_rate \
    --sampling_range $sampling_range \
    --lineage_tokens $lineage_tokens \
    --train_epochs $train_epochs

checkpoint=./outputs/pretrain_checkpoints/Weather/ckpt_best.pth

for pred_len in 96 192 336 720; do
    python -u run.py \
        --task_name fine_tune \
        --is_training 1 \
        --root_path ./dataset/weather/ \
        --data_path weather.csv \
        --model_id weather \
        --model $model_name \
        --data Weather \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --d_layers 1 \
        --n_heads 4 \
        --factor 3 \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --train_epochs 3 \
        --lineage_tokens $lineage_tokens \
        --representation_using $representation_using \
        --load_checkpoints $checkpoint
done


