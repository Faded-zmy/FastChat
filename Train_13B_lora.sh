
#model5 13B lora
deepspeed fastchat/train/train_lora.py \
    --deepspeed deepspeed_config.json \
    --model_name_or_path /ai_efs/models/public_models/baize/baize-13b  \
    --data_path /home/ec2-user/liwei/data/our_data_tov_60k_model5.json \
    --bf16 True \
    --output_dir /ai_efs/models/13B_our_60k_model5_lora \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 400 \
    --save_total_limit 20 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --lazy_preprocess True


