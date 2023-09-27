import os
import json
import numpy as np
import copy

def one_question():
    data_path="/home/liwei/code/FastChat/fastchat/eval/table/review/review_train60k_model3/"
    data_path="/home/ec2-user/liwei/FastChat/fastchat/eval/table/review/review_train60k_13B/"
    #data_path="/home/ec2-user/liwei/FastChat/fastchat/eval/table/review/review_train60k_model3_13B/"

    key=["c{}.jsonl".format(i*200) for i in range(1,7)]+["cfinal.jsonl"]
    for k in key:
        file_path = data_path+k
        with open(file_path, "r") as f:
            json_score = []
            for line in f:
                if np.isnan(json.loads(line)["score"]):
                    continue
                else:
                    json_score.append(json.loads(line)["score"])
        print(k,np.mean(json_score),np.median(json_score),np.std(json_score),np.min(json_score))

def round_question():
    data_path="/home/liwei/code/FastChat/fastchat/eval/table/review/review_train60k_complete/"
    # data_path="/home/ec2-user/liwei/FastChat/fastchat/eval/table/review/review_train60k_model4_7B/"
    # key=["c{}.jsonl".format(i*400) for i in range(5,14)]+["cfinal.jsonl"]

    # data_path="/home/ec2-user/liwei/FastChat/fastchat/eval/table/review/review_train60k_model5_13B/"
    #data_path="/home/ec2-user/liwei/FastChat/fastchat/eval/table/question_and_answer/"
    data_path="/ai_efs/kortex_data/data/model_eval/person4/20230629_reply_lora_B4_L2e-5/"
    key=["c{}.jsonl".format(i*200) for i in range(1,2)] #+["cfinal.jsonl"]
    #key=["review_data{}.jsonl".format(i) for i in [1,2,3,4,5]]
    key=["question{}.jsonl".format(i) for i in [1,2,3]]
    key=["Sherlock_Holmes.jsonl"]
    for k in key:
        file_path = data_path+k
        with open(file_path, "r") as f:
            json_score = {}
            for line in f:
                if np.isnan(json.loads(line)["score"]):
                    continue
                else:
                    question_id=str(json.loads(line)["question_id"]).split("_")
                    cat="_".join(question_id[:-2])
                    question_id=question_id[-2]
                    # if "2" in k or len(question_id)==5:
                    #     cat="_".join(question_id[:-2])
                    #     question_id=question_id[-2]
                    # else:
                    #     cat="_".join(question_id[:-1])
                    #     question_id=question_id[-1]
                    if cat not in json_score:
                        json_score[cat]={}
                    if question_id not in json_score[cat]:
                        json_score[cat][question_id]=[]
                    json_score[cat][question_id].append(json.loads(line)["score"])
        
        all_cat=[]
        for cat in json_score:
            for question_id in json_score[cat]:
                json_score[cat][question_id]=np.mean(json_score[cat][question_id])
            json_score_cat=list(json_score[cat].values())
            all_cat+=json_score_cat
            print(cat,k,np.mean(json_score_cat),np.median(json_score_cat),np.std(json_score_cat),np.min(json_score_cat))

        print("all",k,np.mean(all_cat),np.median(all_cat),np.std(all_cat),np.min(all_cat))

def round_person(data_path,key,json_score_origin):
    # data_path="/ai_efs/kortex_data/data/model_eval/{}/".format(name)
    # #key=["7B_12k_lora_B16_L2e-5.jsonl"]
    # key=os.listdir(data_path)
    for k in key:
        file_path = data_path+k
        with open(file_path, "r") as f:
            json_score = copy.deepcopy(json_score_origin)
            key_list=[i for i in json_score.keys()]
            for line in f:
                results=json.loads(line)["text"]
                score_valid=[]
                score_pair = results.split("\n")
                try:
                    score_valid = [int(s.replace("score: ", "").replace("Score: ", "")) for s in score_pair if 'score:' in s or 'Score:' in s]
                except Exception:
                    continue
                if len(score_valid)!=len(key_list):
                    print(score_pair)
                    continue
                question_id=str(json.loads(line)["question_id"]).split("_")
                if "2" in k or len(question_id)==5:
                    cat="_".join(question_id[:-2])
                    question_id=question_id[-2]
                else:
                    cat="_".join(question_id[:-1])
                    question_id=question_id[-1]
                for i in range(len(score_valid)):
                    key=key_list[i]

                    if cat not in json_score[key]:
                        json_score[key][cat]={}
                    if question_id not in json_score[key][cat]:
                        json_score[key][cat][question_id]=[]
                    json_score[key][cat][question_id].append(score_valid[i])
        score_count=[]
        for key in json_score:
            all_cat=[]
            for cat in json_score[key]:
                for question_id in json_score[key][cat]:
                    json_score[key][cat][question_id]=np.mean(json_score[key][cat][question_id])
                json_score_cat=list(json_score[key][cat].values())
                all_cat+=json_score_cat
            # print(cat,k,np.mean(json_score_cat),np.median(json_score_cat),np.std(json_score_cat),np.min(json_score_cat))
            score_count.append(np.mean(all_cat))
            print(k,key,np.mean(all_cat),np.median(all_cat),np.std(all_cat),np.min(all_cat))
        print(np.mean(score_count))



def count_at():
    data_path="/home/liwei/code/FastChat/fastchat/eval/table/answer/answer_train60k/"

    key=["c{}.jsonl".format(i*200) for i in range(1,8)]
    for k in key:
        file_path = data_path+k
        with open(file_path, "r") as f:
            number,at_number=0,0
            json_score = []
            for line in f:
                number+=1
                if "@" in json.loads(line)["text"]:
                    at_number+=1
        print(number,at_number,at_number/number)


if __name__=="__main__":
    #one_question()
    # round_question()
    #count_at()

    #reply
    # json_score = {"Make sense":{},"Interesting":{},"Insight to character":{},"Personality":{},"Knowledge":{},"Logic":{}}
    # data_path="/ai_efs/kortex_data/data/model_eval/person4/20230629_reply_lora_B4_L2e-5/"
    # key=os.listdir(data_path)
    # round_person(data_path,key,json_score)

    #roomchat
    #ID_info={"ID0": ["Mr. Jobs","Steve Jobs","Steve","Jobs",4],
    ID_info={"ID1": ["Mr. Musk","Elon Musk","Elon","Musk",4],
             "ID2": ["Mr. Gates","Bill Gates","Bill","Gates",4],
             "ID3": ["Mr. Vladimir","Vladimir Putin","Putin","Vladimir",8],
             "ID4": ["Mr. Corleone","Vito Corleone","Vito","Corleone",8],
             "ID5": ["Mr. Gatsby","Jay Gatsby","Jay","Gatsby",4],
             "ID6": ["Mr. Targaryen","Daenerys Targaryen","Daenerys","Targaryen",4],
             "ID7": ["Mr. Potter","Harry Potter","Harry","Potter",4],
             "ID8": ["Mr. Holmes","Sherlock Holmes","Sherlock","Holmes",4],
             "ID9": ["Mr. Drucker","Peter Drucker","Peter","Drucker",4]}
    
    json_score = {"Personality":{},"Knowledge":{},"Logic":{},"Related":{}}
    for id_ in ID_info:
        #for cat in ["reply_roomtchat","roomtchat"]:
        for cat in ["roomtchat"]:
            data_path="/ai_efs/kortex_data/data/model_eval/{}/20230628_{}_lora_B{}_L2e-5/".format(ID_info[id_][1].replace(" ","_"),cat,ID_info[id_][-1])
            key=["roomchat.jsonl"]
            print(data_path)
            round_person(data_path,key,json_score)
