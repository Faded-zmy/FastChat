# 0808
# CUDA_VISIBLE_DEVICES=0  python get_lora_model_answer_round_zmy.py  --model-id vicuna_v1.1 --base-model-path /ai_jfs/models/public_models/baize/baize-7b/  --lora-path /ai_jfs/mengying/model/summary_0809_lora_B16_L2e-5/  --question-file /ai_jfs/mengying/data/sum/eval_question_20230705_gpu2.json --answer-file /ai_jfs/mengying/data/sum/eval_our_model_answer_0814.jsonl  --num-gpus 1

# #llama-2 based
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6  python get_lora_model_answer_round_zmy.py  --model-id vicuna_v1.1 --base-model-path /ai_jfs/models/public_models/llama-2/llama-2-7b-chat/  --lora-path /ai_jfs/mengying/model/summary_0814_lora_B16_L2e-5  --question-file /ai_jfs/mengying/data/sum/eval_question_20230705_gpu2.json --answer-file /ai_jfs/mengying/data/sum/eval_llama2_answer_0814.jsonl  --num-gpus 8 --max-token 4096

#llama-2 based 3quarters
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python get_lora_model_answer_round_zmy.py  --model-id vicuna_v1.1 --base-model-path /ai_jfs/models/public_models/llama-2/llama-2-7b-chat/  --lora-path /ai_jfs/mengying/model/summary_0815_3q_lora_B16_L2e-5  --question-file /ai_jfs/mengying/data/sum/eval_question_20230705_gpu2.json --answer-file /ai_jfs/mengying/data/sum/eval_llama2_3q_answer_0814.jsonl  --num-gpus 8 --max-token 4096

## 加上main_topic这个key
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python get_lora_model_answer_round_zmy.py  --model-id vicuna_v1.1 --base-model-path /ai_jfs/models/public_models/llama-2/llama-2-7b-chat/  --lora-path /ai_jfs/mengying/model/summary_0817_lora_B16_L2e-5  --question-file /ai_jfs/mengying/data/sum/eval_question_20230705_gpu2.json --answer-file /ai_jfs/mengying/data/sum/eval_llama2_main_topic_0817.jsonl  --num-gpus 8 --max-token 4096

# # merge sum
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python get_lora_model_answer_round_zmy.py  --model-id vicuna_v1.1 --base-model-path /ai_jfs/models/public_models/llama-2/llama-2-7b-chat/  --lora-path /ai_jfs/mengying/model/merge_sum_0821_lora_B16_L2e-5  --question-file /ai_jfs/mengying/data/sum/eval_question_20230705_gpu2.json --answer-file /ai_jfs/mengying/data/sum/eval_llama2_main_topic_0817.jsonl  --num-gpus 8 --max-token 4096