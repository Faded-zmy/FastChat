import os
from tqdm import tqdm

if __name__=="__main__":
    ID_info={"ID1": ["Mr. Musk","Elon Musk","Elon","Musk"],
             "ID3": ["Mr. Vladimir","Vladimir Putin","Putin","Vladimir"],
             "ID7": ["Mr. Potter","Harry Potter","Harry","Potter"],
             "ID8": ["Mr. Holmes","Sherlock Holmes","Sherlock","Holmes"]}
    number_gpu=8
    gpus=[str(i) for i in range(number_gpu)]
    for name in ID_info:
        name=ID_info[name][1].replace(" ","_")
        val_path="/ai_efs/kortex_data/data/characters/reply_only/{}_7k_val.json".format(name)
        cmd1="CUDA_VISIBLE_DEVICES={} python get_lora_model_answer_round.py --model-id All --base-model-path /ai_efs/models/public_models/vicuna-7b-v1.1/ --lora-path /ai_efs/models/characters/person4/20230629_reply_lora_B4_L2e-5/ --question-file {} --answer-file /ai_efs/kortex_data/data/model_answer/person4/20230629_reply_lora_B4_L2e-5/{}.jsonl --num-gpus {} --cat Person".format(",".join(gpus),val_path,name,number_gpu)
        print(cmd1)

        os.system(cmd1)
        print("*****************")
        print(name,"roomtchat/roomchat.jsonl down")
        print("*****************")


    #ID_info={"ID0": ["Mr. Jobs","Steve Jobs","Steve","Jobs"],
    ID_info={"ID1": ["Mr. Musk","Elon Musk","Elon","Musk",4],
             "ID2": ["Mr. Gates","Bill Gates","Bill","Gates",4],
             "ID3": ["Mr. Vladimir","Vladimir Putin","Putin","Vladimir",8],
             "ID4": ["Mr. Corleone","Vito Corleone","Vito","Corleone",8],
             "ID5": ["Mr. Gatsby","Jay Gatsby","Jay","Gatsby",4],
             "ID6": ["Mr. Targaryen","Daenerys Targaryen","Daenerys","Targaryen",4],
             "ID7": ["Mr. Potter","Harry Potter","Harry","Potter",4],
             "ID8": ["Mr. Holmes","Sherlock Holmes","Sherlock","Holmes",4],
             "ID9": ["Mr. Drucker","Peter Drucker","Peter","Drucker",4]}
    
    for name in ID_info:
        batch=ID_info[name][-1]
        name=ID_info[name][1].replace(" ","_")
        cmd1="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python get_lora_model_answer_round.py --model-id {} --base-model-path /ai_efs/models/public_models/vicuna-7b-v1.1/ --lora-path /ai_efs/models/characters/{}/20230628_reply_roomtchat_lora_B{}_L2e-5/ --question-file /ai_efs/kortex_data/data/characters/reply_with_prompt/{}_val.json --answer-file /ai_efs/kortex_data/data/model_answer/{}/20230628_reply_roomtchat_lora_B{}_L2e-5/reply.jsonl --num-gpus 8 --cat Person".format(name,name,batch,name,name,batch)
        
        cmd2="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python get_lora_model_answer_round.py --model-id {} --base-model-path /ai_efs/models/public_models/vicuna-7b-v1.1/ --lora-path /ai_efs/models/characters/{}/20230628_reply_roomtchat_lora_B{}_L2e-5/ --question-file /ai_efs/kortex_data/data/characters/roomchat/{}_multi_val.json --answer-file /ai_efs/kortex_data/data/model_answer/{}/20230628_reply_roomtchat_lora_B{}_L2e-5/roomchat.jsonl --num-gpus 8 --cat Roomchat".format(name,name,batch,name,name,batch)
        
        cmd3="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python get_lora_model_answer_round.py --model-id {} --base-model-path /ai_efs/models/public_models/vicuna-7b-v1.1/ --lora-path /ai_efs/models/characters/{}/20230628_roomtchat_lora_B{}_L2e-5/ --question-file /ai_efs/kortex_data/data/characters/roomchat/{}_multi_val.json --answer-file /ai_efs/kortex_data/data/model_answer/{}/20230628_roomtchat_lora_B{}_L2e-5/roomchat.jsonl --num-gpus 8 --cat Roomchat".format(name,name,batch,name,name,batch)

        print(cmd3)
        os.system(cmd3)
        print("*****************")
        print(name,"roomtchat/roomchat.jsonl down")
        print("*****************")

        print(cmd2)
        os.system(cmd2)
        print("*****************")
        print(name,"reply_roomtchat/roomchat.jsonl down")
        print("*****************")

        print(cmd1)
        os.system(cmd1)
        print("*****************")
        print(name,"reply_roomtchat/reply.jsonl down")
        print("*****************")

    
