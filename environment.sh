pip3 install fschat

pip3 install --upgrade pip  # enable PEP 660 support
pip3 install -e .

pip install einops
pip install ninja

sudo yum install centos-release-scl
sudo yum install devtoolset-9-gcc*
scl enable devtoolset-9 bash
pip install flash_attn==1.0.5
pip install deepspeed==0.9.2

pip3 install git+https://github.com/huggingface/peft.git@2822398fbe896f25d4dac5e468624dc5fd65a51b

#torchrun --nproc_per_node=4 --master_port=20001 fastchat/train/train_mem.py \
#    --model_name_or_path /data/home/liwei/model/vicuna-7b-v1.1  \
#    --data_path playground/data/dummy.json \
#    --bf16 True \
#    --output_dir output \
#    --num_train_epochs 3 \
#    --per_device_train_batch_size 2 \
#    --per_device_eval_batch_size 2 \
#    --gradient_accumulation_steps 16 \
#    --evaluation_strategy "no" \
#    --save_strategy "steps" \
#    --save_steps 1200 \
#    --save_total_limit 10 \
#    --learning_rate 2e-5 \
#    --weight_decay 0. \
#    --warmup_ratio 0.03 \
#    --lr_scheduler_type "cosine" \
#    --logging_steps 1 \
#    --fsdp "full_shard auto_wrap" \
#    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
#    --tf32 True \
#    --model_max_length 2048 \
#    --gradient_checkpointing True \
#   --lazy_preprocess True
