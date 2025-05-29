import os
from evaluation.exp_config import *
from inference import query_model
from pathlib import Path


file_path = f"evaluation/eval_points/{MODEL}.txt"
with open(file_path, 'r', encoding='utf-8') as f:
    points = f.read()

dir_path = f'outputs/{MODEL}_repo' # Adjust the path to where your generated code files are located
py_files = sorted(Path(dir_path).glob("*.py"))
code = ""
for file in py_files:
    with open(file, 'r', encoding='utf-8') as f:
        code += f"\n\n# --- File: {file.name} ---\n\n"
        code += f.read()

prompt = f"""Now, I'm presenting you with a generated code. You need to check whether the details of the code correspond to the key points.
The experiment instruction for the generated code is
{INSTRUCTION}
you just need to consider model, task and dataset used in the instruction.
There are a total of 5 comparison points, and each point is scored from 0 to 20. A score of 20 indicates perfect reproduction, while 0 means no reproduction at all.
Please rate each of the 5 comparison points separately and provide the reasons.
Points to compare
{points}
Genetaerated code
{code}
For each scoring criterion, you need to give a score between 0-20 points. The scoring criteria are as follows:
Total difference: 0-2 points.
Unsimilar: 3-8 points.
Similar: 9-14 points.
Roughly the same: 15-18 points.
Same: 19-20 points.
You need to conduct the evaluation with a critical perspective. Don't give high scores to the mismatched content. You need to disregard all comments, as they do not pertain to the implementation of the code.
The code needs to be strictly corresponding to the points. Any mismatched content should result in deduction of points.
The scores should be presented in the form of [x/20 points]. The final score should be the sum of the scores for each point.
"""
resp = query_model(model_str='o1', prompt=prompt, temp=0.6)


os.makedirs(f"evaluation/eval_results/high", exist_ok=True)
with open(f"evaluation/eval_results/high/{MODEL}.txt", "w", encoding="utf-8") as f:
    f.write(resp)