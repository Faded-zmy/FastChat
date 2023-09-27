#5model lora
# deepspeed fastchat/train/train_lora.py \
#     --deepspeed deepspeed_config.json \
#     --model_name_or_path /ai_efs/models/public_models/baize/baize-7b  \
#     --data_path /home/ec2-user/liwei/data/our_data_tov_60k_all_5model.json \
#     --bf16 True \
#     --output_dir /ai_efs/models/7B_our_60k_all_5model_lora \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 2 \
#     --gradient_accumulation_steps 16 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 1000 \
#     --save_total_limit 40 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing False \
#     --lazy_preprocess True

#5model 4e-5
# deepspeed fastchat/train/train_lora.py \
#     --deepspeed deepspeed_config.json \
#     --model_name_or_path /ai_efs/models/public_models/baize/baize-7b  \
#     --data_path /home/ec2-user/liwei/data/our_data_tov_60k_all_5model.json \
#     --bf16 True \
#     --output_dir /ai_efs/models/7B_our_60k_all_5model_lora \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 2 \
#     --gradient_accumulation_steps 16 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 1000 \
#     --save_total_limit 40 \
#     --learning_rate 4e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing False \
#     --lazy_preprocess True

#push jd
# CUDA_VISIBLE_DEVICES=4,5,6,7  deepspeed fastchat/train/train_lora.py \
#     --deepspeed deepspeed_config.json \
#     --model_name_or_path /ai_efs/models/public_models/baize/baize-7b  \
#     --data_path /home/ec2-user/liwei/data/our_data_tov_60k_all.json \
#     --bf16 True \
#     --output_dir /ai_efs/models/7B_our_60k_smmarization123_lora_B16_L2e-5 \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 2 \
#     --gradient_accumulation_steps 16 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 1000 \
#     --save_total_limit 40 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing False \
#     --lazy_preprocess True

#reply
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 deepspeed fastchat/train/train_lora.py \
    --deepspeed deepspeed_config.json \
    --model_name_or_path /ai_jfs/models/public_models/baize/baize-7b  \
    --data_path /ai_jfs/mengying/data/sum/train_data_0809.json \
    --bf16 True \
    --output_dir /ai_jfs/mengying/model/card \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
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
