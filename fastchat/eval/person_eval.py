import os
from tqdm import tqdm

if __name__=="__main__":
    #ID_info={"ID1": ["Mr. Musk","Elon Musk","Elon","Musk",4],
    ID_info={"ID2": ["Mr. Gates","Bill Gates","Bill","Gates",4],
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
        for cat in ["reply_roomtchat","roomtchat"]:
            cmd="python eval_gpt_review_person.py -name {} -a /ai_efs/kortex_data/data/model_answer/{}/20230628_{}_lora_B{}_L2e-5/roomchat.jsonl -p table/prompt_our_all.jsonl -r table/reviewer_our.jsonl -o /ai_efs/kortex_data/data/model_eval/{}/20230628_{}_lora_B{}_L2e-5/roomchat.jsonl".format(name,name,cat,batch,name,cat,batch)
            os.system(cmd)

            print("******************")
            print(name,"roomchat eval down")
            print("******************")
    
    


    

    
