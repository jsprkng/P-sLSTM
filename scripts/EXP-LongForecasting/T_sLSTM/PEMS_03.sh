# T_sLSTM: the P_sLSTM settings for this dataset, unchanged, plus a Transformer
# encoder (defaults --t_layers 2 --t_heads 4). Pass --t_layers 0 to reproduce P_sLSTM.
# add --individual for P-sLSTM
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

seq_len=96

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS03.npz \
  --model_id PEMS03_$seq_len'_'12 \
  --model T_sLSTM \
  --data PEMS \
  --features M \
  --seq_len $seq_len \
  --pred_len 12 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.001 \
  --patch_size 16 --stride 16 \
  --num_blocks 2 \
  --channel 358 --embedding_dim 300 --num_heads 6 --conv1d_kernel_size 8 --group_norm_weight True \
  --dropout 0.1 --patience 3 >logs/LongForecasting/T_sLSTM_PEMS03_$seq_len'_'12.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS03.npz \
  --model_id PEMS03_$seq_len'_'24 \
  --model T_sLSTM \
  --data PEMS \
  --features M \
  --seq_len $seq_len \
  --pred_len 24 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.001 \
  --patch_size 16 --stride 16 \
  --num_blocks 2 \
  --channel 358 --embedding_dim 300 --num_heads 6 --conv1d_kernel_size 8 --group_norm_weight True \
  --dropout 0.1 --patience 3 >logs/LongForecasting/T_sLSTM_PEMS03_$seq_len'_'24.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS03.npz \
  --model_id PEMS03_$seq_len'_'48 \
  --model T_sLSTM \
  --data PEMS \
  --features M \
  --seq_len $seq_len \
  --pred_len 48 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.001 \
  --patch_size 16 --stride 16 \
  --num_blocks 2 \
  --channel 358 --embedding_dim 200 --num_heads 2 --conv1d_kernel_size 8 --group_norm_weight True \
  --dropout 0.1 --patience 3 >logs/LongForecasting/T_sLSTM_PEMS03_$seq_len'_'48.log


python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS03.npz \
  --model_id PEMS03_$seq_len'_'96 \
  --model T_sLSTM \
  --data PEMS \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.001 \
  --patch_size 16 --stride 16 \
  --num_blocks 2 \
  --channel 358 --embedding_dim 300 --num_heads 6 --conv1d_kernel_size 8 --group_norm_weight True \
  --dropout 0.1 --patience 3 >logs/LongForecasting/T_sLSTM_PEMS03_$seq_len'_'96.log
