from evaluation.exp_config import *
from pathlib import Path
from inference import query_model
import re
import os


gt_path = f"reproducebench/{MODEL}/run_{MODEL}.py"
with open(gt_path, 'r', encoding='utf-8') as f:
    ground_truth_code = f.read()

file_path = f"evaluation/eval_points/{MODEL}.txt"
with open(file_path, 'r', encoding='utf-8') as f:
    points = f.read()

dir_path = f'outputs/{MODEL}_repo' # Adjust the path to where your generated code files are located
py_files = Path(dir_path).glob("*.py")
generated_code = ""
for file in py_files:
    print(file)
    with open(file, 'r', encoding='utf-8') as f:
        generated_code += f"\n\n# --- File: {file.name} ---\n\n"
        generated_code += f.read()

generated_code = re.sub(r'(""".*?"""|\'\'\'.*?\'\'\')', '', generated_code, flags=re.DOTALL)

prompt = f"""
You are an expert AI code reviewer specializing in evaluating the fidelity of research paper implementations. Your task is to meticulously compare a `generated_code` against a set of `key_points` derived from a research paper. Crucially, you will use the provided `reference_code` (the official or canonical implementation from the paper's authors) as the ground truth to understand how each `key_point` is specifically and correctly implemented.
The experiment instruction is
{INSTRUCTION}
you just need to consider model, task and dataset used in the instruction.
**Inputs You Will Be Provided With:**

1.  `points`: A list of key concepts, mechanisms, algorithms, or architectural features from the research paper that the `generated_code` is supposed to implement.
2.  `reference_code`: The official source code accompanying the research paper. This code serves as the benchmark for understanding the precise, intended implementation details of each `key_point`.
3.  `generated_code`: The `generated_code` that needs to be evaluated for its accuracy in reproducing the `key_points` as they are implemented in the `reference_code`.

**Your Evaluation Process (to be performed for EACH `key_point` listed in `points`):**

1.  **Understand Key Point via Reference Code:**
    * For the current `key_point`, first, thoroughly examine the `reference_code`.
    * Identify and describe the specific segment(s) of the `reference_code` (e.g., functions, classes, logic blocks) that implement this `key_point`.
    * Summarize how the `reference_code` realizes this `key_point`. This understanding will be your basis for comparison.

2.  **Analyze Generated Code against Reference Implementation:**
    * Now, review the `generated_code` (`generated_code`) to find its implementation of the same `key_point`.
    * Compare this implementation directly against your understanding of how it was done in the `reference_code`. Focus on whether the logic, structure, and functional outcome are equivalent.

3.  **Score the Replication:**
    * Based on your comparative analysis, assign a score from 0 to 20 to the `generated_code` for its replication of this specific `key_point`, using the scoring rubric below.

4.  **Provide Detailed Justification:**
    * Clearly articulate the reasons for your score.
    * Specifically highlight matches and discrepancies between the `generated_code`'s implementation and the `reference_code`'s implementation of the `key_point`. Explain *why* it matches or *why* it deviates.

**Scoring Rubric (applied individually to each `key_point`):**

0-2 points (Total difference): The core innovation point (as demonstrated in the reference_code) is not replicated at all in the generated_code, or the implementation is fundamentally flawed, missing major functional aspects, or entirely incorrect when compared to the reference_code.
3-8 points (Unsimilar): The key_point is replicated in the generated_code but not completely accurately or comprehensively when compared to the reference_code. Some aspects of the reference_code's implementation might be present, but there are noticeable inaccuracies, missing details, or differences in logic that might affect functionality or deviate from the paper's intended mechanism as shown in the official code.
9-14 points (Similar): The key_point is replicated completely and accurately in the generated_code. The implementation in the generated_code closely mirrors the logic, structure, and functional behavior of the corresponding implementation in the reference_code, although there might be some non-critical differences or alternative approaches that achieve the same core outcome.
15-18 points (Roughly the same): The key_point is replicated completely, accurately, and comprehensively in the generated_code. The implementation is highly consistent with the reference_code in logic, structure, and function, differing only in very minor, non-functional ways (e.g., variable naming, comments, slight code formatting) that do not impact the core mechanism.
19-20 points (Same): The implementation of the key_point in the generated_code is identical or functionally equivalent to the reference_code, representing a code-level copy or a near-perfect replication of the relevant sections.

**Critical Evaluation Guidelines:**

Conduct your evaluation with a stringent and critical perspective. Do not award high scores for superficial similarities or implementations that do not match the essence of the `reference_code`'s approach for a given `key_point`.
Your evaluation must be based solely on the executable code logic. Disregard all comments in both the `reference_code` and the `generated_code` as they do not pertain to the functional implementation.
At the same time, you need to pay attention not only to the key points themselves, but also to all the details related to them. If the key points correspond but the related implementations are different, it will still affect the reproduction effect.
Importantly, you need to focus on the specific implementation of the code rather than its overall structure. For instance, if the key points are designed for upsampling, you should first analyze the specific implementation of upsampling in the reference code (e.g. nn.ConvTranspose2d or Interpolation), and then analyze how it is implemented in the generated code. However, the module implemenations are much important than common practice such as normalization and activation functions.
Please evaluate each key point in the `points` with the generated code. You should both determine the overall similarity and concrete scores. For example, when you decide the two codes are similar, you also should determine the level of similar.
When evaluating specific implementations, prioritize the equivalence of core structure and function over superficial differences, such as the number of modules. If generated code achieves the same functional outcome and structural design as the reference, it should be considered equivalent, irrespective of modular composition.
**Begin Evaluation:**

Points to compare:
{points}

Reference Code:
{ground_truth_code}

Generated code:
{generated_code}

**Output Format:**

For each `key_point`, please provide:
1.  **Key Point:** [Name/Description of the key point being evaluated]
2.  **Reference Code Implementation Summary:** [Your summary of how this key point is implemented in the `reference_code`]
3.  **Generated Code Analysis & Comparison:** [Your detailed analysis of the `generated_code`'s attempt to implement this point, comparing it directly to the `reference_code`'s approach]
4.  **Score:** [x/20 points]
5.  **Reasoning for Score:** [Detailed justification based on the comparison]

Sum the overall scores for each `key_point` to provide a final score out of 100 points, and include a summary of the overall evaluation.
**Overall Score:** [x/100 points]
"""


import os
resp = query_model(model_str='o1', system_prompt='', prompt=prompt, temp=0.6)
os.makedirs(f"evaluation/eval_results/mixed", exist_ok=True)

with open(f"evaluation/eval_results/mixed/{MODEL}.txt", "w", encoding="utf-8") as f:
    f.write(resp)