import argparse
import os
import re
from evaluation.exp_config import *
from inference import query_model
from pathlib import Path


def eval_low_level(paper_name, ground_truth_code_dir="reproducebench", generated_code_dir="outputs", 
                              output_results_dir="evaluation/eval_results/low", judge_model="o1"):

    # Construct path to ground truth code
    gt_path = os.path.join(ground_truth_code_dir, paper_name, f"run_{paper_name}.py")
    try:
        with open(gt_path, 'r', encoding='utf-8') as f:
            ground_truth_code = f.read()
    except FileNotFoundError:
        print(f"Error: Ground truth code file not found at '{gt_path}'")
        return
    except Exception as e:
        print(f"Error reading ground truth code file '{gt_path}': {e}")
        return

    # Construct path to generated code directory
    generated_code_dir = os.path.join(generated_code_dir, f"{paper_name}")
    if not os.path.isdir(generated_code_dir):
        print(f"Error: Generated code directory not found at '{generated_code_dir}'")
        return

    # Read content from all .py files in the generated code directory
    py_files = sorted(Path(generated_code_dir).glob("*.py"))
    generated_code = ""
    if not py_files:
        print(f"No Python files found in '{generated_code_dir}' to evaluate.")
        return

    for file in py_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                # Remove triple-quoted strings (docstrings/multi-line comments)
                file_content = re.sub(r'(""".*?"""|\'\'\'.*?\'\'\')', '', f.read(), flags=re.DOTALL)
                generated_code += f"\n\n# --- File: {file.name} ---\n\n"
                generated_code += file_content
        except Exception as e:
            print(f"Error reading generated code file '{file}': {e}")
            continue

    # Construct the prompt for evaluation
    prompt = f"""
You are an AI expert proficient in code analysis, specializing in comparing and evaluating code structure and functionality. Now you need to judge whether the replicated code is the same as the official code.
Please analyze the following two code segments: the first is model-generated code, and the second is standard training code. You don't need to pay attention to the contents related to saving, printing and logging. Focus on the model itself and its training process instead.
Conclusion: Summarize whether the two code segments are completely equivalent in terms of model definition and Experimental Integrity, and briefly explain the significance of the scoring results.

First Code Segment (Standard Training Code):\n{ground_truth_code}

Second Code Segment (Model-Generated Code):\n{generated_code}

Your task is to conduct a detailed comparison of these two code segments, focusing on the following aspects:

Overall Structure:
The overall structure of the model code, including the data data flow in the forward function and the overall model structure. 
Only consider the model-related code, such as the encoder-decoder structure, and ignore the data loading and other irrelevant code.

Model Structure:
Are the model architectures defined in both code segments (e.g., number of layers, activation functions, input/output dimensions) completely identical?
Specifically compare the implementation details, such as if a spatio-temporal processing module is present in the code, including but not limited to input processing, feature extraction methods, temporal dimension handling, spatial dimension handling, connection methods (e.g., residual connections, attention mechanisms), and parameter settings. Apply this level of comparison to all modules.
Check for any subtle differences (e.g., convolution kernel size, pooling methods, normalization techniques).

Training Details:
Compare whether the model hyperparameters (e.g., learning rate, batch size, optimizer type, learning rate decay strategy) are consistent.
Verify whether the loss function’s definition and implementation are identical, including the loss calculation formula and weight assignments.
Experimental Integrity:
Compare the implementation of the training process, including the training pipeline, data preprocessing, and gradient update logic, to determine if they are equivalent. The most crucial thing you need to pay attention to is the integrity of the experiment.
Check for any functional differences (e.g., initialization methods, early stopping mechanisms). There is no requirement for checkpoint saving, logging information and multi-gpu training. Do not consider these contents.

Notes:
Pay special attention to analyzing the implementation of the module in the model structure, ensuring a detailed comparison of each layer’s specific parameters and computational logic.
If differences are found, clearly indicate the specific code lines or modules where they occur and analyze their potential impact on model performance or behavior.
Ignore differences in code style (e.g., variable naming, comments) and focus on functionality and implementation logic.
You need to focus on the code implementation and don't need to consider comments and function names and other contents irrelevant to the implementation.

Scoring Criteria (Total: 100 points):
Overall Structure (25 points): Consistency in the overall structure of the model realted code, including the overall data processing pipeline in forward function, the overall modules structured, for example encoder->decoder structures.
Model Details (25 points): Consistency in the implementation of the model structure. There are many modules in the model, all of them should be compared. 
If the names are different but the internal functions are the same, you should consider them as the same. 
Training Details (25 points): Consistency in hyperparameters, loss function, learning rate decay, etc.
Experimental Integrity (25 points): Consistency in training loop, data processing, and other pipeline. You just need to compare the training, testing pipeline included in the code.
The final score is the sum of the three components, rounded to the nearest integer.
You need to analyze the code details and implementation before giving the score. Firstly, analyze the overall structure of the model and then specific to each module.
For custom blocks, the details of how custom blocks is implemented should be analyzed.
For some official implementation, you need to analyze whether the implementation of custom is the same as the official one based on your understanding. Ignoring the differences of programming languages and only considering the code implementation
For each scoring criterion, you need to give a score between 0-25 points. The scoring criteria are as follows:
Total difference: 0-4 points.\nUnsimilar: 5-10 points.\nSimilar: 12-16 points.\nRoughly the same: 17-22 points.\nSame: 23-25 points.
For example, ```python coarse_up = F.interpolate(coarse, size=fine.shape[-2:], mode='bilinear', align_corners=True) ``` and  ```python self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)\nx1 = self.up(x1)``` should be considered the same as they both perform upsampling with bilinear interpolation.
The MLP (Multi-Layer Perceptron) and single linear layer, you should be consider as roughly the same. 
If the difference between DoubleConv and nn.Conv lies in the fact that DoubleConv is a more complex structure, you should consider them as roughly the same.
Simlar should be when the functions of two codes are the same, but the implementation are different.
Unsimilar should be when the functions of two codes are different.
If one code directly import model structure from package and the other code implements the model structure by itself, you should evaluate the implementation of two model structure specifically, consider whether the implementation is the similar or not.
You need to analyze the specific implementation of the code, do not just focus on the name, the . You need to make detailed evaluations based on the details. Ignore all the comments and focus only on the code.

Notes:
In Training Details: Do not consider the chekpoint saving, logging information, multi-gpu training, validation and test frequency and best model saving.

Output Format:
Similarities: List the parts where the model structure, training details, and Experimental Integrity are completely consistent.
Differences: List any differences found, including the specific code sections, a description of the differences, and their potential impact.
Scoring:
Overall Structure: XX/25 (explain the basis for the score)
Model Details: XX/25 (explain the basis for the score)
Training Details: XX/25 (explain the basis for the score)
Experimental Integrity: XX/25 (explain the basis for the score)
Total Score: XX/100
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
    parser = argparse.ArgumentParser(description="Evaluate generated code against ground truth for paper replication.")
    
    parser.add_argument('--paper_name', type=str, default=MODEL)
    parser.add_argument('--ground_truth_code_dir', type=str, default="reproducebench")
    parser.add_argument('--generated_code_dir', type=str, default="outputs")
    parser.add_argument('--output_results_dir', type=str, default="evaluation/eval_results/low")
    parser.add_argument('--judge_model', type=str, default='o1')

    args = parser.parse_args()

    # Call the function with parsed arguments
    eval_low_level(
        paper_name=args.paper_name,
        ground_truth_code_dir=args.ground_truth_code_dir,
        generated_code_dir=args.generated_code_dir,
        output_results_dir=args.output_results_dir,
        judge_model=args.judge_model
    )