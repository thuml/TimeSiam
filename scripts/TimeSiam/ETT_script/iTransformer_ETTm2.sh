export CUDA_VISIBLE_DEVICES=0
model_name=iTransformer
train_epochs=50
mask_rate=0.25
pretrain_seq_len=96
sampling_range=12
lineage_tokens=6
representation_using=concat

python -u run.py \
    --task_name timesiam \
    --is_training 0 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --model_id ETTm2 \
    --model $model_name \
    --data ETTm2 \
    --features M \
    --seq_len $pretrain_seq_len \
    --e_layers 1 \
    --d_layers 1 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 256 \
    --d_ff 256 \
    --mask_rate $mask_rate \
    --sampling_range $sampling_range \
    --lineage_tokens $lineage_tokens \
    --train_epochs $train_epochs

checkpoint=./outputs/pretrain_checkpoints/ETTm2/ckpt_best.pth


for pred_len in 96 192 336 720; do
    python -u run.py \
        --task_name fine_tune \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTm2.csv \
        --model_id ETTm2 \
        --model $model_name \
        --data ETTm2 \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 1 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --d_model 256 \
        --d_ff 256 \
        --lineage_tokens $lineage_tokens \
        --representation_using $representation_using \
        --load_checkpoints $checkpoint \
        --head_dropout 0.2
done


