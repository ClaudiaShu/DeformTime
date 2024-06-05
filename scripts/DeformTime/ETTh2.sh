export CUDA_VISIBLE_DEVICES=1

model_name=DeformTime


python -u run.py \
  --root_path ./dataset/ETT/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96 \
  --model $model_name \
  --data ETTh2 \
  --features MS \
  --seq_len 336 \
  --pred_len 96 \
  --label_len 0 \
  --batch_size 16 \
  --d_model 16 \
  --learning_rate 0.0002 \
  --train_epochs 100 \
  --dropout 0 \
  --layer_dropout 0 \
  --e_layers 2 \
  --d_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --n_heads 4 \
  --n_reshape 24 \
  --patch_len 24 \
  --kernel 7 \
  --patience 5 \
  --des 'Exp' \
  --itr 1


python -u run.py \
  --root_path ./dataset/ETT/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_192 \
  --model $model_name \
  --data ETTh2 \
  --features MS \
  --seq_len 336 \
  --pred_len 192 \
  --label_len 0 \
  --batch_size 32 \
  --d_model 64 \
  --learning_rate 0.0002 \
  --train_epochs 100 \
  --dropout 0 \
  --layer_dropout 0 \
  --e_layers 2 \
  --d_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --n_heads 4 \
  --n_reshape 6 \
  --patch_len 12 \
  --kernel 5 \
  --patience 5 \
  --des 'Exp' \
  --itr 1


python -u run.py \
  --root_path ./dataset/ETT/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_336 \
  --model $model_name \
  --data ETTh2 \
  --features MS \
  --seq_len 336 \
  --pred_len 336 \
  --label_len 0 \
  --batch_size 32 \
  --d_model 64 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --dropout 0 \
  --layer_dropout 0 \
  --e_layers 2 \
  --d_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --n_heads 4 \
  --n_reshape 6 \
  --patch_len 24 \
  --kernel 3 \
  --patience 5 \
  --des 'Exp' \
  --itr 1


python -u run.py \
  --root_path ./dataset/ETT/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_720 \
  --model $model_name \
  --data ETTh2 \
  --features MS \
  --seq_len 336 \
  --pred_len 720 \
  --label_len 0 \
  --batch_size 32 \
  --d_model 16 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --dropout 0 \
  --layer_dropout 0 \
  --e_layers 2 \
  --d_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --n_heads 4 \
  --n_reshape 24 \
  --patch_len 6 \
  --kernel 9 \
  --patience 5 \
  --des 'Exp' \
  --itr 1