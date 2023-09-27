import argparse
import json
import os
import time
import numpy as np

import openai
import tqdm
import ray

import shortuuid
import logging

from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_API_RETRY = 5
REQ_TIME_GAP = 10


@ray.remote(num_cpus=10)
def get_eval(answer_jsons,reviewer_jsons,prompt_jsons,name,max_tokens,output_review_file):
    question_list=range(len(answer_jsons)) #[:1]
    for i in tqdm(question_list):
        ques = answer_jsons[i]["question"]
        cat = answer_jsons[i]["cat"]
        ans1 = answer_jsons[i]["text"]
        sys_prompt, prompt, reviewer_id = gen_prompt(
            reviewer_jsons, prompt_jsons, cat, ques, ans1, name
        )

        content="None"
        for k in range(MAX_API_RETRY):
            openai.api_key = "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" #"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"
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    temperature=0.2,  # TODO: figure out which temperature is best for evaluation
                    max_tokens=max_tokens,
                )
                content = response["choices"][0]["message"]["content"]
                break
            except Exception as e:
                logger.error(e)
                time.sleep(2)
        
        scores = parse_score(content)
        review_id = shortuuid.uuid()
        with open(os.path.expanduser(output_review_file), "a") as fout:
            ans_json={
                    "review_id": review_id,
                    "question_id": answer_jsons[i]["question_id"],
                    "answer1_id": answer_jsons[i]["answer_id"],
                    "reviewer_id": reviewer_id,
                    "metadata": {},
                    "text":content,
                    "score":scores
                }
            fout.write(json.dumps(ans_json) + "\n")
        time.sleep(REQ_TIME_GAP)

@ray.remote(num_cpus=4)
def try_gpt_4(sys_prompt, user_prompt: str, max_tokens: int):
    openai.api_type = "azure"
    openai.api_base = "https://novatech.openai.azure.com/"
    openai.api_version = "2023-03-15-preview"
    openai.api_key = "34b9886fbce94d9db0ec4445455196f3"
    response = openai.ChatCompletion.create(
        engine="gpt-4-32k",
        messages = [
            {"role": "system", "content": sys_prompt},
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
            ],
        temperature=0.9,
        max_tokens=max_tokens,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)
    content = response['choices'][0]['message']['content']
    logger.info(content)
    return content

def parse_score(review):
    try:
        score_pair = review.split("\n")
        score_valid = [int(s.replace("score: ", "").replace("Score: ", "")) for s in score_pair if 'score:' in s or 'Score:' in s]
        return np.mean(score_valid)
    except Exception as e:
        logger.error(
            f"{e}\nContent: {review}\n" "You must manually fix the score pair."
        )
        return 0


def gen_prompt(reviewer_jsons, prompt_jsons, cat, ques, ans1, name=""):
    # Default to general category (index=0)
    reviewer_idx = 0
    for idx, reviewer in enumerate(reviewer_jsons):
        if reviewer["category"] == cat:
            reviewer_idx = idx
            break
    #认为是general任务
    # print(cat)
    prompt_id = reviewer_jsons[reviewer_idx]["prompt_id"]
    prompt_json = prompt_jsons[prompt_id]
    # print(reviewer_idx)
    assert prompt_json["prompt_id"] == prompt_id

    sys_prompt = prompt_json["system_prompt"]
    prompt_template = prompt_json["prompt_template"]
    defaults = prompt_json["defaults"]
    if name!="":
        prompt = prompt_template.format(
            question=ques.replace("an artificial intelligence assistant",name.replace("_"," ")), answer=ans1, **defaults
        )
    else:
        prompt = prompt_template.format(
            question=ques, answer=ans1, **defaults
        )
    # print(prompt)

    return sys_prompt, prompt, reviewer_idx + 1


def get_json_list(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, "r") as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line))
        return json_list

def run_eval_gpt35(name,answer_file_list,reviewer_file,prompt_file,output_review_file,model_eval,max_tokens,num_cpus):
    
    if not os.path.exists(os.path.dirname(output_review_file)):
        os.makedirs(os.path.dirname(output_review_file))

    with open(os.path.expanduser(output_review_file), "w") as fout:
            ans_json={
                    "question_id": "None",
                }
            fout.write(json.dumps(ans_json) + "\n")
    
    answer_jsons = get_json_list(answer_file_list)
    reviewer_jsons = get_json_list(reviewer_file)
    prompt_jsons = get_json_list(prompt_file)

    handles = []
    review_jsons = []
    
    chunk_size = len(answer_jsons) // num_cpus
    
    for i in range(0,len(answer_jsons),chunk_size):
        if model_eval=="gpt4":
            handles.append(
                try_gpt_4.remote(
                    answer_jsons[i : i + chunk_size],
                    reviewer_jsons,
                    prompt_jsons,
                    name,
                    max_tokens,
                    output_review_file
                )
            )
        else:
            handles.append(
                get_eval.remote(
                    answer_jsons[i : i + chunk_size],
                    reviewer_jsons,
                    prompt_jsons,
                    name,
                    max_tokens,
                    output_review_file
                )
            )

    ray.get(handles)


    # question_idx_list = list(range(total_len))

    # reviews = ray.get(handles)
    # scores_all=[]
    # with open(f"{output_review_file}", "w") as output_review_file:
    #     for idx, review in enumerate(reviews):
    #         scores = parse_score(review)
    #         scores_all.append(scores)
    #         review_jsons[idx]["text"] = review
    #         review_jsons[idx]["score"] = scores
    #         output_review_file.write(json.dumps(review_jsons[idx]) + "\n")

    # print(np.mean(scores_all))

def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            if qid=="None":continue
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatGPT-based QA evaluation.")
    parser.add_argument("-name", "--name",default="")
    parser.add_argument("-a", "--answer-file-list")
    parser.add_argument("-p", "--prompt-file")
    parser.add_argument("-r", "--reviewer-file")
    parser.add_argument("-o", "--output-review-file")
    parser.add_argument("-m", "--model-eval",default="gpt35")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="maximum number of tokens produced in the output",
    )
    parser.add_argument("-n", "--num-cpus",default=10)
    args = parser.parse_args()

    ray.init()

    run_eval_gpt35(args.name,
                    args.answer_file_list,
                    args.reviewer_file,
                    args.prompt_file,
                    args.output_review_file,
                    args.model_eval,
                    args.max_tokens,
                    int(args.num_cpus)
                    )

    reorg_answer_file(args.output_review_file)