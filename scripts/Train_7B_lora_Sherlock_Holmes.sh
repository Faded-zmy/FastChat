CUDA_VISIBLE_DEVICES=2,3  deepspeed --master_port=29500 fastchat/train/train_lora.py \
    --deepspeed deepspeed_config.json \
    --model_name_or_path /ai_efs/models/public_models/vicuna-7b-v1.1/  \
    --data_path /ai_efs/kortex_data/data/characters/roomchat_reply/Sherlock_Holmes_train.json \
    --bf16 True \
    --output_dir /ai_efs/models/characters/Sherlock_Holmes/20230628_reply_roomtchat_lora_B4_L2e-5 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 40 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --lazy_preprocess True 