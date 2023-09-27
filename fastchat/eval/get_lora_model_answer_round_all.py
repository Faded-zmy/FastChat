import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
from peft import PeftModel
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import ray

from fastchat.conversation import get_conv_template, SeparatorStyle
from fastchat.utils import disable_torch_init


def run_eval(lora_path, base_model_path, model_id, question_file, answer_file, num_gpus, cat):
    # split question file into num_gpus files
    ques_jsons = json.load(open(question_file,"r"))

    # ques_jsons=ques_jsons[:1]
    chunk_size = len(ques_jsons) // num_gpus
    ans_handles = []
    for i in range(0, len(ques_jsons), chunk_size):
        ans_handles.append(
            get_model_answers.remote(
                lora_path,
                base_model_path, 
                model_id, 
                ques_jsons[i : i + chunk_size],
                answer_file, 
                cat
            )
        )

    ray.get(ans_handles)

    # with open(os.path.expanduser(answer_file), "w") as ans_file:
    #     for line in ans_jsons:
    #         ans_file.write(json.dumps(line) + "\n")

@ray.remote(num_gpus=1)
@torch.inference_mode()
def get_model_answers(lora_path, base_model_path, model_id, question_jsons, answer_file, cat):
    disable_torch_init()
    if "baichuan" in base_model_path:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False,trust_remote_code=True)
        #tokenizer.pad_token_id = 0 if tokenizer.pad_token_id == 64000 else tokenizer.pad_token_id
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype=torch.float16,trust_remote_code=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype=torch.float16
        )

    lora_model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        torch_dtype=torch.float16,
    )

    model = lora_model.merge_and_unload().cuda()
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)

    ans_jsons = []
    for i, ques_json in enumerate(tqdm(question_jsons)):
        idx=ques_json["id"]
        conversations=ques_json["conversations"]
        if model_id=="All" and "name" in ques_json:
            model_id = ques_json[name]
        conv = get_conv_template(model_id).copy()
        outputs_conv=[]
        # conv = get_default_conv_template(model_id).copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        if roles[conversations[0]["from"]]!= conv.roles[0]:
            conversition=conversition[1:]
        for j, sentence in tqdm(enumerate(conversations)):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            if "baichuan" in base_model_path and sentence["from"]=="gpt":
                conv.append_message(role, "<s> "+sentence["value"])
            else:
                conv.append_message(role, sentence["value"])
            if j % 2 == 0:
                conv_eval = conv.copy()
                conv_eval.append_message(conv.roles[1], None)
                assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO
                prompt = conv_eval.get_prompt()
                if "baichuan" in base_model_path:
                    prompt += "<s> "
                
                # print("***********")
                # print(prompt)
                # print("***********")

                input_ids = tokenizer([prompt]).input_ids
                output_ids = model.generate(
                    torch.as_tensor(input_ids).cuda(),
                    do_sample=True,
                    temperature=0.7,
                    max_new_tokens=2048,
                )
                output_ids = output_ids[0][len(input_ids[0]) :]
                outputs = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                outputs_conv.append(outputs)


                # print("***********************")
                # print("outputs",outputs)
                # print("***********************")

                with open(os.path.expanduser(answer_file), "a") as fout:
                    ans_json = {
                                "question_id": idx,
                                "text": outputs,
                                "answer_id": shortuuid.uuid(),
                                "model_id": model_id,
                                "metadata": {},
                                "question":prompt,
                                "cat":cat
                            }
                    fout.write(json.dumps(ans_json) + "\n")

def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--lora-path", type=str, required=True)
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answer-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--cat", type=str, default="Person")

    args = parser.parse_args()

    ray.init()
    run_eval(
        args.lora_path,
        args.base_model_path,
        args.model_id,
        args.question_file,
        args.answer_file,
        args.num_gpus,
        args.cat
    )
    
    reorg_answer_file(args.answer_file)

