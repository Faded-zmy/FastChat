"""then review results with gpt3.5"""
import argparse
import json
import os
import csv
import time
import concurrent.futures
import ray
import torch

import openai
import tqdm
import shortuuid

from fastchat.conversation import get_default_conv_template, compute_skip_echo_len
from fastchat.utils import disable_torch_init
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM

# MODEL = "gpt-3.5-turbo"
# MODEL_ID = "gpt-3.5-turbo:20230327"

# openai.api_key = "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"

def get_candicates_answers(questions_path: str):

    questions_list = []
    answers_list=[]

    with open(questions_path,encoding='utf-8-sig') as f:
        for row in csv.reader(f,skipinitialspace=True):
            question_id = row[0]
            question = row[1]
            answer = row[2]

            questions_list.append(question)
            answers_list.append(answer)

    return questions_list, answers_list

@ray.remote(num_gpus=1)
@torch.inference_mode()
def get_model_answers(questions_path, model_path, model_id, num_gpus):
    ques_jsons = []
    with open(os.path.expanduser(questions_path), "r") as ques_file:
        for line in ques_file:
            ques_jsons.append(line)

    chunk_size = len(ques_jsons) // num_gpus
    ans_handles = []
    for i in range(0, len(ques_jsons), chunk_size):
        ans_handles.append(
            get_batch_answers.remote(
                model_path, model_id, ques_jsons[i : i + chunk_size]
            )
        )
    

    ans_jsons = []
    for ans_handle in ans_handles:
        ans_jsons.extend(ray.get(ans_handle))

    with open(os.path.expanduser(answer_file), "w") as ans_file:
        for line in ans_jsons:
            ans_file.write(json.dumps(line) + "\n")

    with open(os.path.expanduser(args.questions_path)) as f:
        for line in f:
            if not line:
                continue
            q = json.loads(line)
            questions_dict[q["question_id"]] = q["text"]

    answers_dict={}
    with open(os.path.expanduser(args.answers_path)) as f:
        for line in f:
            if not line:
                continue
            q = json.loads(line)
            answers_dict[q["question_id"]] = q["text"]

    return questions_dict, answers_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatGPT answer generation and compare.")
    parser.add_argument("--question",default="", required=True)
    parser.add_argument("--answer",default="")
    parser.add_argument("--model_path",default="")
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="maximum number of tokens produced in the output",
    )
    args = parser.parse_args()

    #如果是面试结果，那就是一个csv读取
    if args.question == args.answer:
        questions_list,answers_list = get_candicates_answers(args.question)
    else: #如果是我们自己的模型结果，那就要生成以后再获取了
        questions_dict,answers_dict = get_model_answers(args.question, args.model_path, args.model_id, args.num_gpus)

    reviewer_jsons

