from evaluation.exp_config import *
from inference import query_model

file_path = f"evaluation/papers/{MODEL}.txt"
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

prompt = f"""
The provided text above is the full text of the paper. The instruction of reproducing the experiemnt is {INSTRUCTION} with the model {MODEL}.
Now you need to evaluate whether the code has replicated the instruction about {TASK} experiments in the paper. 
Please summarize 5 key points. These 5 key points will be used to assess whether the code has completely replicated the model, methods, and experiments setting in the paper. 
Specifically, you need to use 3 points to summarize the model method, 1 for hyperparameters and 1 for training setup is recommanded. Do not include the dataset generation process as a point, as the dataset has been preprocessed.
You should regard each part of the proposed method in the paper as a separate key point. 
If there are formulas in the paper, you need to extract them in LaTeX format and use them as the criteria for judging whether the code has been replicated. 
Do not include some common contents as the key points to be compared. Only include the key points related to the {TASK} task. 
Do summaeize the data generation process and various model structures as the key points, the instruction and dataloader code have contained specific model and dataset.
If all these 5 points are replicated exactly, the code will fully replicate the paper. These key points are very important and should be as detailed as possible, which could reflect the key points to reproduce the paper."
"""

resp = query_model(model_str='o1', system_prompt='', prompt=prompt+'\n\n'+content, temp=0.6)
with open(f"evaluation/eval_points/{MODEL}.txt", "w", encoding="utf-8") as f:
    f.write(resp)