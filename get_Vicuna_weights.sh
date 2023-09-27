#change vicuna
# python -m fastchat.model.apply_delta \
#     --base-model-path /data/home/liwei/model/pyllama_data/7B_convert/ \
#     --target-model-path /data/home/liwei/model/vicuna-7b-v1.1 \
#     --delta-path /data/home/liwei/model/vicuna-7b-delta-v1.1


# python -m fastchat.model.apply_delta \
#     --base-model-path /data/home/liwei/model/pyllama_data/13B_convert/ \
#     --target-model-path /data/home/liwei/model/vicuna-13b-v1.1 \
#     --delta-path /data/home/liwei/model/vicuna-13b-delta-v1.1

#change baize
# python3 -m fastchat.model.apply_lora \
#         --base /ai_efs/models/public_models/pyllama/7B_convert \
#         --target /ai_efs/models/public_models/baize/baize-7b \
#         --lora /ai_efs/models/public_models/baize/baize-lora-7B

# # #change baize
# python3 -m fastchat.model.apply_lora \
#         --base /ai_efs/models/public_models/pyllama/7B_convert \
#         --target /ai_efs/models/public_models/baize/baize-7b \
#         --lora /ai_efs/models/public_models/baize/baize-lora-7B

# python3 -m fastchat.model.apply_lora \
#         --base /ai_efs/models/public_models/pyllama/13B_convert \
#         --target /ai_efs/models/public_models/baize/baize-13b \
#         --lora /ai_efs/models/public_models/baize/baize-lora-13B

# #30B
# python3 -m fastchat.model.apply_lora \
#         --base /ai_efs/models/public_models/pyllama/30B_convert \
#         --target /ai_efs/models/public_models/baize/baize-30b \
#         --lora /ai_efs/models/public_models/baize/baize-lora-30B

# #change train_baize
# python3 -m fastchat.model.apply_lora \
#         --base /ai_efs/models/public_models/baize/baize-13b \jiush
#         --target /ai_efs/models/13B_our_60k_model5_lora_llm \
#         --lora /ai_efs/models/13B_our_60k_model5_lora

# #change train_baize
# python3 -m fastchat.model.apply_lora \
#         --base /ai_efs/models/public_models/baize/baize-7b \
#         --target /ai_efs/models/7B_our_60k_all_5model_lora_llm \
#         --lora /ai_efs/models/7B_our_60k_all_5model_lora


# #change train_baize
# python3 -m fastchat.model.apply_lora \
#         --base /ai_efs/models/public_models/baize/baize-7b \
#         --target /ai_efs/models/7B_our_60k_all_5model_lora_B16_L4e-5_llm \
#         --lora /ai_efs/models/7B_our_60k_all_5model_lora_B16_L4e-5

# #change train_baize
# python3 -m fastchat.model.apply_lora \
#         --base /ai_efs/models/public_models/baize/baize-13b \
#         --target /ai_efs/models/13B_our_60k_all_model5_lora_B16_L1e-5_llm \
#         --lora /ai_efs/models/13B_our_60k_all_model5_lora_B16_L1e-5

#python3 -m fastchat.model.apply_lora \
#        --base /ai_efs/models/public_models/baize/baize-13b \
#        --target /ai_efs/models/13B_our_60k_all_model5_lora_llm \
#        --lora /ai_efs/models/13B_our_60k_all_model5_lora
#
# python3 -m fastchat.model.apply_lora \
#         --base /ai_efs/models/public_models/baize/baize-13b \
#         --target /ai_efs/models/13B_our_60k_split_lora_B16_L2e-5_llm \
#         --lora /ai_efs/models/13B_our_60k_split_lora_B16_L2e-5

# python3 -m fastchat.model.apply_lora \
#         --base /ai_efs/models/public_models/baize/baize-7b \
#         --target /ai_efs/models/7B_our_60k_split_lora_B16_L2e-5_llm \
#         --lora /ai_efs/models/7B_our_60k_split_lora_B16_L2e-5

# python3 -m fastchat.model.apply_lora \
#         --base /ai_efs/models/public_models/baize/baize-13b \
#         --target /ai_efs/models/13B_our_60k_split_lora_B16_L1e-5_llm \
#         --lora /ai_efs/models/13B_our_60k_split_lora_B16_L1e-5

# python3 -m fastchat.model.apply_lora \
#         --base /ai_efs/models/public_models/baize/baize-7b \
#         --target /ai_efs/models/7B_our_60k_smmarization123_lora_B16_L2e-5_llm \
#         --lora /ai_efs/models/7B_our_60k_smmarization123_lora_B16_L2e-5

python3 -m fastchat.model.apply_lora \
        --base /ai_efs/models/public_models/baize/baize-13b \
        --target /ai_efs/models/13B_our_60k_smmarization123_lora_B16_L1e-5_llm \
        --lora /ai_efs/models/13B_our_60k_smmarization123_lora_B16_L1e-5