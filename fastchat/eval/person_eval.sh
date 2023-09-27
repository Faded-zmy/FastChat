python eval_gpt_review_person.py -name Harry_Potter \
        -a /ai_efs/kortex_data/data/model_answer/person4/20230629_reply_lora_B4_L2e-5/Harry_Potter.jsonl \
        -p table/prompt_our_all.jsonl \
        -r table/reviewer_our.jsonl \
        -o /ai_efs/kortex_data/data/model_eval/person4/20230629_reply_lora_B4_L2e-5/Harry_Potter.jsonl

python eval_gpt_review_person.py -name Sherlock_Holmes \
        -a /ai_efs/kortex_data/data/model_answer/person4/20230629_reply_lora_B4_L2e-5/Sherlock_Holmes.jsonl \
        -p table/prompt_our_all.jsonl \
        -r table/reviewer_our.jsonl \
        -o /ai_efs/kortex_data/data/model_eval/person4/20230629_reply_lora_B4_L2e-5/Sherlock_Holmes.jsonl

python eval_gpt_review_person.py -name Elon_Musk \
        -a /ai_efs/kortex_data/data/model_answer/person4/20230629_reply_lora_B4_L2e-5/Elon_Musk.jsonl \
        -p table/prompt_our_all.jsonl \
        -r table/reviewer_our.jsonl \
        -o /ai_efs/kortex_data/data/model_eval/person4/20230629_reply_lora_B4_L2e-5/Elon_Musk.jsonl