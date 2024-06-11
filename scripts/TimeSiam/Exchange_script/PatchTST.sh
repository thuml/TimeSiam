export CUDA_VISIBLE_DEVICES=0
model_name=PatchTST
train_epochs=50
mask_rate=0.25
pretrain_seq_len=96
sampling_range=6
#lineage_tokens=2

representation_using=avg

python -u run.py \
    --task_name timesiam \
    --is_training 0 \
    --root_path ./dataset/exchange_rate/ \
    --data_path exchange_rate.csv \
    --model_id Exchange \
    --model $model_name \
    --data Exchange \
    --features M \
    --seq_len $pretrain_seq_len \
    --e_layers 2 \
    --enc_in 8 \
    --dec_in 8 \
    --c_out 8 \
    --mask_rate $mask_rate \
    --sampling_range $sampling_range \
    --train_epochs $train_epochs

checkpoint=./outputs/pretrain_checkpoints/Exchange/ckpt_best.pth

for pred_len in 96 192 336 720; do
    python -u run.py \
        --task_name fine_tune \
        --is_training 1 \
        --root_path ./dataset/exchange_rate/ \
        --data_path exchange_rate.csv \
        --model_id Exchange \
        --model $model_name \
        --data Exchange \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 8 \
        --dec_in 8 \
        --c_out 8 \
        --representation_using $representation_using \
        --load_checkpoints $checkpoint \
        --batch_size 64 \
        --head_dropout 0
done