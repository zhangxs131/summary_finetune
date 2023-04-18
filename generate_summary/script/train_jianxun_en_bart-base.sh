python train.py \
    --model_name_or_path ../pretrain_model/bart-base \
    --train_file ../data_now/train.json \
    --validation_file ../data_now/dev.json \
    --output_dir output/ \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --num_beams=3 \
    --min_summ_length=100 \
    --max_summ_length=250 \
    --length_penalty=1.0 \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4