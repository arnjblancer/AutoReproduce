import argparse
import os
from evaluation.exp_config import *
from inference import query_model


def summarize_points(paper_name, task, instruction="reproduce the given task in the paper", paper_dir="evaluation/papers", 
                     output_dir="evaluation/eval_points", api_model_str="o1"):

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct file paths
    paper_path = os.path.join(paper_dir, f"{paper_name}.txt")
    output_path = os.path.join(output_dir, f"{paper_name}.txt")

    # Read content from the paper file
    try:
        with open(paper_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: Paper text file not found at '{paper_path}'")
        return ""
    except Exception as e:
        print(f"Error reading paper text file '{paper_path}': {e}")
        return ""

    # Construct the prompt
    prompt_text = f"""
The provided text above is the full text of the paper. The instruction of reproducing the experiment is {instruction} with the model {paper_name}.
Now you need to evaluate whether the code has replicated the instruction about {task} experiments in the paper.
Please summarize 5 key points. These 5 key points will be used to assess whether the code has completely replicated the model, methods, and experiments setting in the paper.
Specifically, you need to use 3 points to summarize the model method, 1 for hyperparameters and 1 for training setup is recommanded. Do not include the dataset generation process as a point, as the dataset has been preprocessed.
You should regard each part of the proposed method in the paper as a separate key point.
If there are formulas in the paper, you need to extract them in LaTeX format and use them as the criteria for judging whether the code has been replicated.
Do not include some common contents as the key points to be compared. Only include the key points related to the {task} task.
Do summaeize the data generation process and various model structures as the key points, the instruction and dataloader code have contained specific model and dataset.
If all these 5 points are replicated exactly, the code will fully replicate the paper. These key points are very important and should be as detailed as possible, which could reflect the key points to reproduce the paper."
"""

    # Query the AI model and save response
    try:
        resp = query_model(
            model_str=api_model_str,
            system_prompt='',
            prompt=prompt_text + '\n\n' + content,
            temp=0.6
        )
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(resp)
        print(f"Evaluation points saved to '{output_path}'")
        return output_path
    except Exception as e:
        print(f"Error querying the model or saving response: {e}")
        return ""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evaluation points for paper replication.")
    
    parser.add_argument('--paper_dir', type=str, default="evaluation/papers")
    parser.add_argument('--output_dir', type=str, default="evaluation/eval_points")
    parser.add_argument('--paper_name', type=str, default=MODEL)
    parser.add_argument('--task', type=str, default=TASK)
    parser.add_argument('--instruction', type=str, default=INSTRUCTION)
    parser.add_argument('--api_model_str', type=str, default='o1')

    args = parser.parse_args()

    # Call the function with parsed arguments
    summarize_points(
        paper_name=args.paper_name,
        task=args.task,
        instruction=args.instruction,
        paper_dir=args.paper_dir,
        output_dir=args.output_dir,
        api_model_str=args.api_model_str
    )