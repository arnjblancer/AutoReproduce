import argparse
import os
from evaluation.exp_config import *
from inference import query_model
from pathlib import Path


def eval_high_level(paper_name, instruction="reproduce the given task in the paper", eval_points_dir="evaluation/eval_points", generated_code_dir="outputs", 
                         output_results_dir="evaluation/eval_results/high", judge_model="o1"):

    # Construct path to evaluation points file
    eval_points_path = os.path.join(eval_points_dir, f"{paper_name}.txt")
    try:
        with open(eval_points_path, 'r', encoding='utf-8') as f:
            points = f.read()
    except FileNotFoundError:
        print(f"Error: Evaluation points file not found at '{eval_points_path}'")
        return
    except Exception as e:
        print(f"Error reading evaluation points file '{eval_points_path}': {e}")
        return

    # Construct path to generated code directory
    generated_code_dir = os.path.join(generated_code_dir, f"{paper_name}")
    if not os.path.isdir(generated_code_dir):
        print(f"Error: Generated code directory not found at '{generated_code_dir}'")
        return

    # Read content from all .py files in the generated code directory
    py_files = sorted(Path(generated_code_dir).glob("*.py"))
    code = ""
    if not py_files:
        print(f"No Python files found in '{generated_code_dir}' to evaluate.")
        return

    for file in py_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                code += f"\n\n# --- File: {file.name} ---\n\n"
                code += f.read()
        except Exception as e:
            print(f"Error reading generated code file '{file}': {e}")
            continue

    # Construct the prompt for evaluation
    prompt = f"""Now, I'm presenting you with a generated code. You need to check whether the details of the code correspond to the key points.
The experiment instruction for the generated code is\n{instruction}
you just need to consider model, task and dataset used in the instruction.
There are a total of 5 comparison points, and each point is scored from 0 to 20. A score of 20 indicates perfect reproduction, while 0 means no reproduction at all.
Please rate each of the 5 comparison points separately and provide the reasons.
Points to compare\n{points}\nGenerated code\n{code}
For each scoring criterion, you need to give a score between 0-20 points. The scoring criteria are as follows:
Total difference: 0-2 points. Unsimilar: 3-8 points. Similar: 9-14 points. Roughly the same: 15-18 points. Same: 19-20 points.
You need to conduct the evaluation with a critical perspective. Don't give high scores to the mismatched content. You need to disregard all comments, as they do not pertain to the implementation of the code.
The code needs to be strictly corresponding to the points. Any mismatched content should result in deduction of points.
The scores should be presented in the form of [x/20 points]. The final score should be the sum of the scores for each point.
"""
    # Query the AI model and save response
    try:
        resp = query_model(
            model_str=judge_model,
            prompt=prompt,
            temp=0.6
        )
        os.makedirs(output_results_dir, exist_ok=True)
        output_path = os.path.join(output_results_dir, f"{paper_name}.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(resp)
        print(f"Evaluation results saved to '{output_path}'")
    except Exception as e:
        print(f"Error querying the model or saving response: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated code against key points for paper replication.")
    
    parser.add_argument('--paper_name', type=str, default=MODEL)
    parser.add_argument('--instruction', type=str, default=INSTRUCTION)
    parser.add_argument('--eval_points_dir', type=str, default="evaluation/eval_points")
    parser.add_argument('--generated_code_dir', type=str, default="outputs")
    parser.add_argument('--output_results_dir', type=str, default="evaluation/eval_results/high")
    parser.add_argument('--judge_model', type=str, default='o1')

    args = parser.parse_args()

    # Call the function with parsed arguments
    eval_high_level(
        paper_name=args.paper_name,
        instruction=args.instruction,
        eval_points_dir=args.eval_points_dir,
        generated_code_dir=args.generated_code_dir,
        output_results_dir=args.output_results_dir,
        judge_model=args.judge_model
    )