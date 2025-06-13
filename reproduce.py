from agents import *
from copy import copy
from common_imports import *
import ast
import argparse
import pickle
from evaluation.exp_config import ARXIV_ID, TASK, TITLE, INSTRUCTION, MODEL
import time
DEFAULT_LLM_BACKBONE = "o3-mini"

class AutoReproduce:
    def __init__(self, arxiv_id, paper_path, instruction, dataloader_code, max_steps=20, num_papers_lit_review=1, agent_model_backbone=f"{DEFAULT_LLM_BACKBONE}", notes=list(), human_in_loop_flag=None, compile_pdf=True, mlesolver_max_steps=3, papersolver_max_steps=5):

        self.notes = notes
        self.max_steps = max_steps
        self.compile_pdf = compile_pdf
        self.arxiv_id = arxiv_id
        self.paper_path = paper_path
        self.instruction = instruction
        self.dataloader_code = dataloader_code
        self.model_backbone = agent_model_backbone
        self.num_papers_lit_review = num_papers_lit_review
        self.instruction = instruction
        self.print_cost = True
        self.reference_papers = list()

        self.arxiv_num_summaries = 5

        self.phases = [
            ("literature review", ["overall summary", "method summary", "experiment summary"]),
            # ("paper lineage", ["paper lineage"]),
            ("experimentation", ["data acquisition", "method replication", "experiment execution", "code refactor"]),
        ]
        self.phase_status = dict()
        for phase, subtasks in self.phases:
            for subtask in subtasks:
                self.phase_status[subtask] = False

        self.phase_models = dict()
        if type(agent_model_backbone) == str:
            for phase, subtasks in self.phases:
                for subtask in subtasks:
                    self.phase_models[subtask] = agent_model_backbone

        elif type(agent_model_backbone) == dict:
            self.phase_models = agent_model_backbone


        self.human_in_loop_flag = human_in_loop_flag

        self.statistics_per_phase = {
            "overall summary":      {"time": 0.0, "steps": 0.0,},
            "method summary":       {"time": 0.0, "steps": 0.0,},
            "experiment summary":   {"time": 0.0, "steps": 0.0,},
            "paper lineage":        {"time": 0.0, "steps": 0.0,},
            "data acquisition":     {"time": 0.0, "steps": 0.0,},
            "method replication":   {"time": 0.0, "steps": 0.0,},
            "experiment execution": {"time": 0.0, "steps": 0.0,},
            "code refactor":        {"time": 0.0, "steps": 0.0,},
        }

        self.save = True
        self.verbose = True
        self.codeagent = CodeAgent(model=self.model_backbone, notes=self.notes, max_steps=self.max_steps, dataloader_code=self.dataloader_code, task_instruction=self.instruction)
        self.researchagent = ResearchAgent(model=self.model_backbone, notes=self.notes, max_steps=self.max_steps, task_instruction=self.instruction)


        remove_directory("output_dir")
        # make dirs
        os.makedirs(os.path.join(".", "output_dir"), exist_ok=True)
        os.makedirs(os.path.join("output_dir", "save_stats"), exist_ok=True)
        os.makedirs(os.path.join("output_dir", "ckpts"), exist_ok=True)

    def set_model(self, model, agent='both'):
        self.set_agent_attr("model", model, agent)

    def save_state(self, phase):
        phase = phase.replace(" ", "_")

        with open(f"output_dir/save_stats/{phase}.pkl", "wb") as f:
            pickle.dump(self, f)

    def set_agent_attr(self, attr, obj, agent='both'):
        if agent=='both':
            setattr(self.researchagent, attr, obj)
            setattr(self.codeagent, attr, obj)
        elif agent=='code':
            setattr(self.codeagent, attr, obj)
        elif agent=='research':
            setattr(self.researchagent, attr, obj)

    def reset_agents(self):
        self.researchagent.reset()
        self.codeagent.reset()

    def perform_research(self):
        self.set_model(self.model_backbone)
        for phase, subtasks in self.phases:
            phase_start_time = time.time()  # Start timing the phase
            if self.verbose: print(f"{'*'*50}\nBeginning phase: {phase}\n{'*'*50}")
            for subtask in subtasks:
                if self.verbose: print(f"{'&'*30}\nBeginning subtask: {subtask}\n{'&'*30}")
                
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "overall summary":
                    repeat = True
                    while repeat: repeat = self.overall_summary()
                    self.phase_status[subtask] = True
                    print('Overall Summary Finished !!!!')
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "method summary":
                    repeat = True
                    while repeat: repeat = self.method_summary()
                    self.phase_status[subtask] = True
                    print('Method Summary Finished !!!!')
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "experiment summary":
                    repeat = True
                    while repeat: repeat = self.experiment_summary()
                    self.phase_status[subtask] = True
                    print('Experiment Summary Finished !!!!')
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "paper lineage":
                    repeat = True
                    while repeat: repeat = self.paper_lineage()
                    self.phase_status[subtask] = True
                    print('Paper Lineage Finished !!!!')
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "data acquisition":
                    repeat = True
                    while repeat: repeat = self.data_acquisition()
                    self.phase_status[subtask] = True
                    print('Data Acquisition Finished !!!!')
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "method replication":
                    repeat = True
                    while repeat: repeat = self.method_replication()
                    self.phase_status[subtask] = True
                    print('Method Replication Finished !!!!')
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "experiment execution":
                    repeat = True
                    while repeat: repeat = self.experiment_execution()
                    self.phase_status[subtask] = True
                    print('Experiment Execution Finished !!!!')
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "code refactor":
                    repeat = True
                    while repeat: repeat = self.code_refactor()
                    self.phase_status[subtask] = True
                    print('Code Refactor Finished !!!!')
                if self.save: self.save_state(subtask)
                # Calculate and print the duration of the phase
                phase_end_time = time.time()
                phase_duration = phase_end_time - phase_start_time
                print(f"Subtask '{subtask}' completed in {phase_duration:.2f} seconds.")
                self.statistics_per_phase[subtask]["time"] = phase_duration

    def code_refactor(self):
        info_ = f'Code that needs to be organized\n{self.codeagent.prev_code}. Only generate one code snippet in ```python\n``` format, without the file name'
        resp = self.codeagent.inference(self.instruction, "code refactoring", info=info_, step=0, use_command=False)
        complete_code = extract_python_code(resp)
        save_to_file("output_dir/ckpts", "main.py", complete_code)
        self.reset_agents()

    def experiment_execution(self):
        max_tries = self.max_steps
        ra_feedback = str()
        feedback = str()
        info_ = str()
        related_repo = list(reversed([repr(item["review"]+item["related_code"]) for item in self.codeagent.code]))
        repo_code = self.codeagent.format_relate_repo(related_repo)
        
        info_ = f'Baseline works implmentations\n{repo_code}The code previous code base that you could be used\n{self.codeagent.prev_code}. Your generated code should be concated with previous code.\n'
        resp = self.codeagent.inference(self.instruction, "experiment execution", info=info_, step=0, use_command=False)
        initial_code = extract_python_code(resp)
        print(initial_code)
        new_code = self.codeagent.prev_code + '\n\n' + initial_code
        code_resp, code_error = execute_code(new_code, timeout=300000)
        save_to_file("output_dir/ckpts", "run_v0.py", new_code)
        code_list, code_line = generate_code_lines(new_code)
        feedback = f"\nCode Response\n{code_resp}\n"
        if code_error is not None:
            print('ERROR', code_error)
            info_ = f"The experiment summary of the source paper\n{self.researchagent.paper[0]['experiment_summary']}\n{new_code}\n{code_error}"
            info_ += '\nDetermine whether the error is caused by a bug or if it is indeed due to certain training setting. Give debugging suggestions and the reasons for the error.'
            analysis = self.codeagent.inference(self.instruction, "error analysis", info=info_, step=0, use_command=False)
            print('ANALYSIS', analysis)
            feedback += f"\n{code_error}\nMake modifications based on the analysis.\n{analysis}"
            run_success = False
        else:
            run_success = True

        for _i in range(1, max_tries):
            feedback_ = f'\nThe previous code base\n{code_line}\nEdit the previous code to fix bugs or meet feedbacks\n{feedback}{ra_feedback}'
            if run_success:
                info_ = f"Baseline works implmentations:\n{repo_code}Paper summary:\n{self.researchagent.paper[0]['summary']}+{self.researchagent.paper[0]['experiment_summary']}"
            resp = self.codeagent.inference(self.instruction, "experiment execution", feedback=feedback_, info=info_, step=_i)
            if "```EDIT" in resp:
                edit_code = extract_prompt(resp, "EDIT").split("\n")
                print('EDIT\n', edit_code)
                lines_to_edit = edit_code[0].split(" ")
                if len(lines_to_edit) != 2:
                    feedback = "The format of Edit command is not valid. You should follow. ```EDIT N M\n<new lines to replace old lines>\n``` where output N M two lines."
                    continue
                lines_to_edit = [int(_) for _ in lines_to_edit]
                new_code = edit_code_line(lines_to_edit[0], lines_to_edit[1], code_list, edit_code[1:])
                code_list, code_line = generate_code_lines(new_code)
                code_resp, code_error = execute_code(new_code, timeout=30000)
                feedback = f"\nCode Response\n{code_resp}\n"

                if code_error is not None:
                    print('ERROR\n', code_error)
                    info_ = f'{new_code}\nERROR:{code_error}'
                    analysis = self.codeagent.inference(self.instruction, "error analysis", info=info_, step=0, use_command=False)
                    print('ANALYSIS\n', analysis)
                    feedback += f"{code_error}\nYou must address and fix this error first.\n{analysis}"
                    save_to_file("output_dir/ckpts", f"run_v{_i}_fail.py", new_code)
                    continue
                else:
                    save_to_file("output_dir/ckpts", f"run_v{_i}_success.py", new_code)
            else:
                continue

            info_ = f"Replicated experiment summary:\n{self.researchagent.paper[0]['experiment_summary']}\n\nThe full experiment code\n{new_code}"
            resp = self.researchagent.inference(self.instruction, "experiment execution", info=info_, step=_i, use_command=False)
            compare_instruct = f"COMPARE:\n{resp}"
            print(compare_instruct)
            valid_command = False
            
            while not valid_command:
                feedback_ = f"Code-summary comparison:\n{resp}. You should choose to ```SUBMIT``` or ```UPDATE``` accoring to the comparison."
                resp = self.researchagent.inference(self.instruction, "experiment execution", feedback=f"{feedback_}", step=_i)
              
                if "```UPDATE" in resp:
                    inform_content = extract_prompt(resp, "UPDATE")
                    print("UPDATE\n", inform_content)
                    info_ = f"The paper full text\n{self.researchagent.paper[0]['full_text']}\naCurrent summary\n{self.researchagent.paper[0]['experiment_summary']}\n"
                    info_ += f'Code-summary comparison:\n{compare_instruct}\n{inform_content}. In this phase, you just need to compare the method related components, the code is not need train at this stage.'
                    resp = self.researchagent.inference(self.instruction, "update", info=info_, step=_i, use_command=False)
                    ra_feedback = f"{resp}\nFollowing the information to edit the code.\n{resp}"
                    valid_command = True
                
                elif "```SUBMIT" in resp or "SUBMIT" in resp:
                    print("SUBMIT!!!")
                    save_to_file("output_dir/ckpts", "run.py", new_code)
                    self.reset_agents()
                    self.statistics_per_phase["experiment execution"]["steps"] = _i
                    self.set_agent_attr("prev_code", new_code)
                    valid_command = True
                    return False
                else:
                    print("The response is not valid. You should choose to ```SUBMIT``` or ```UPDATE``` accoring to the comparison.")

    def method_replication(self):
        max_tries = self.max_steps
        ra_feedback = str()
        feedback = str()
        info_ = str()
        related_repo = list(reversed([repr(item["related_code"]) for item in self.codeagent.code]))
        repo_code = self.codeagent.format_relate_repo(related_repo)

        info_ = f"Baseline works implmentations\n{repo_code}\n{self.researchagent.paper[0]['summary']}\n{self.researchagent.paper[0]['method_summary']}\nThe dataloader code\n```python\n{self.codeagent.prev_code}```"
        info_ += '\nYou need to generate the model code for replicating the method summary. Only generate one code snippet in ```python\n``` format, without the file name.'
        resp = self.codeagent.inference(self.instruction, "method replication", info=info_, step=0, use_command=False)
        initial_code = extract_python_code(resp)

        print(initial_code)
        new_code = self.codeagent.prev_code + '\n\n' + initial_code
        code_resp, code_error = execute_code(new_code, timeout=300)
        save_to_file("output_dir/ckpts", f"model_v0.py", new_code)
        code_list, code_line = generate_code_lines(new_code)
        feedback = f"\nCode Response: {code_resp}\n"

        if code_error is not None:
            print('ERROR', code_error)
            info_ = f"The method summary of the source paper\n{self.researchagent.paper[0]['method_summary']}\n{new_code}\n{code_error}"
            info_ += '\nDetermine whether the error is caused by a bug or if it is indeed due to certain model structures. Give debugging suggestions and the reasons for the error.'
            analysis = self.codeagent.inference(self.instruction, "error analysis", info=info_, step=0, use_command=False)
            print('ANALYSIS', analysis)
            feedback += f"\n{code_error}\nMake modifications based on the analysis.\n{analysis}"
            run_success = False
        else:
            run_success = True
        
        for _i in range(1, max_tries):
            feedback_ = f'\nThe previous code base\n{code_line}\nEdit the previous code to fix bugs or meet feedbacks\n{feedback}{ra_feedback}'
            if run_success:
                info_ = f"Baseline works implmentations:\n{repo_code}Paper summary:\n{self.researchagent.paper[0]['summary']+self.researchagent.paper[0]['method_summary']}"
            resp = self.codeagent.inference(self.instruction, "method replication", feedback=feedback_, info=info_, step=_i)

            if "```EDIT" in resp:
                edit_code = extract_prompt(resp, "EDIT").split("\n")
                print('EDIT\n', edit_code)
                snippet_list = get_content_with_number_start(edit_code)
                edits = []
                for code in snippet_list:
                    lines_to_edit = code[0].strip().split(" ")
                    if len(lines_to_edit) != 2:
                        print("Invalid format:", code[0])
                        continue
                    start, end = [int(_) for _ in lines_to_edit]
                    edits.append((start, end, code[1:]))
                edits.sort(reverse=True)
                for start, end, new_lines in edits:
                    new_code = edit_code_line(start, end, code_list, new_lines)
                    code_list, code_line = generate_code_lines(new_code)
                print(f'Edit {len(edits)} snippets')
                code_resp, code_error = execute_code(new_code, timeout=300)
                if self.verbose: print(f"\nCODE RESPONSE:\n{code_resp}")
                feedback = f"\nCode Response: {code_resp}\n"

                if code_error is not None:
                    print('ERROR', code_error)
                    info_ = f'{new_code}\nERROR:{code_error}'
                    analysis = self.codeagent.inference(self.instruction, "error analysis", info=info_, step=0, use_command=False)
                    print('ANALYSIS', analysis)
                    bash_code = extract_bash_code(analysis)
                    if bash_code != '':
                        subprocess.run([bash_code], capture_output=True, text=True)
                    feedback += f"{code_error}\nMake modifications based on the analysis.\n{analysis}"
                    save_to_file("output_dir/ckpts", f"method_v{_i}_fail.py", new_code)
                    run_success = False
                    continue
                else:
                    save_to_file("output_dir/ckpts", f"method_v{_i}_success.py", new_code)
                    run_success = True

            else:
                feedback = "IMPORTANT!!! Don not generate answers directly, you need to generate specific commands ```EDIT N M\n<new code>\n``` commmand."
                continue
            
            info_ = f"Replicated paper summary:\n{self.researchagent.paper[0]['method_summary']}\nThe model code generated\n{new_code}"
            info_ += "Importantly, in order to ensure that the code is bug-free, you need to check whether the code has passed the data validation and whether the entire process is correct."
            resp = self.researchagent.inference(self.instruction, "method replication", info=info_, step=_i, use_command=False)
            compare_instruct = f"COMPARE\n{resp}"
            print(f"COMPARE\n{resp}")
            valid_command = False

            while not valid_command:
                feedback_ = f"Code-summary comparison:\n{resp}. You should choose to ```SUBMIT``` or ```UPDATE``` accoring to the comparison."
                resp = self.researchagent.inference(self.instruction, "method replication", feedback=f"{feedback_}", step=_i)
                if "```UPDATE" in resp:
                    update_content = extract_prompt(resp, "UPDATE")
                    info_ = f"The paper full text\n{self.researchagent.paper[0]['full_text']}\naCurrent summary\n{self.researchagent.paper[0]['method_summary']}\n"
                    info_ += f'Code-summary comparison:\n{compare_instruct}\n{update_content}. In this phase, you just need to compare the method related components, the code is not need train at this stage.'
                    resp = self.researchagent.inference(self.instruction, "update", info=info_, step=_i, use_command=False)
                    print('UPDATE\n', )
                    ra_feedback = f"{resp}\nFollowing the information to edit the code.\n{resp}"
                    valid_command = True
                elif "```SUBMIT" in resp or "SUBMIT" in resp:
                    print("SUBMIT!!!")
                    save_to_file("output_dir/ckpts", "model.py", new_code)
                    self.reset_agents()
                    self.statistics_per_phase["method replication"]["steps"] = _i
                    self.set_agent_attr("prev_code", new_code)
                    valid_command = True
                    return False
                else:
                    print("The response is not valid. You should choose to ```SUBMIT``` or ```UPDATE``` accoring to the comparison.")

    def data_acquisition(self):
        max_tries = self.max_steps
        is_checked = False
        dataloader_code = self.codeagent.dataloader_code
        if dataloader_code is None:
            dataloader_code = ''
            info_ = 'No dataloader code is provided, please generate the dataloader code according to the instructions. The code should be wrapped in ```python\n```'
        else:
            code_list, code_line = generate_code_lines(dataloader_code)
            info_ = f'The code base that you could be used {code_line}.'
        resp = self.codeagent.inference(self.instruction, "data acquisition", info=info_, step=0, use_command=False)
        initial_code = extract_python_code(resp)
        print('Initial Code\n', initial_code)
        new_code = dataloader_code + '\n\n' + initial_code
        code_resp, code_error = execute_code(new_code, timeout=300)
        save_to_file("output_dir/ckpts", f"data_v0.py", new_code)

        code_list, code_line = generate_code_lines(new_code)
        feedback = f"\nCode Response: {code_resp}\n"

        if code_error is not None:
            print('ERROR', code_error)
            feedback += f"\n{code_error}\nERROR: Code had an error! You must address and fix this error first."

        for _i in range(1, max_tries):
            feedback_ = f'\nThe previous code base\n{code_line}\nYou should edit the previous code to fix bugs or meet feedbacks\n{feedback}'
            resp = self.codeagent.inference(self.instruction, "data acquisition", feedback=feedback_, step=_i)
            if "```EDIT" in resp:
                edit_code = extract_prompt(resp, "EDIT").split("\n")
                lines_to_edit = edit_code[0].split(" ")
                print("EDIT\n", edit_code)
                if len(lines_to_edit) == 2:
                    lines_to_edit = [int(_) for _ in lines_to_edit]
                    new_code = edit_code_line(lines_to_edit[0], lines_to_edit[1], code_list, edit_code[1:])
                code_list, code_line = generate_code_lines(new_code)
                print('EDIT\n', edit_code)
                code_resp, code_error = execute_code(new_code, timeout=300)
                if self.verbose: print("!"*100, "\n", f"CODE RESPONSE: {code_resp}")

                feedback = f"\nCode Response: {code_resp}\n"
                if code_error is not None:
                    print('ERROR\n', code_error)
                    feedback += f"\n{code_error}\nERROR: Code had an error! You must address and fix this error first."
                    save_to_file("output_dir/ckpts", f"data_v{_i}_fail.py", new_code)
                    continue
                else:
                    save_to_file("output_dir/ckpts", f"data_v{_i}_success.py", new_code)
                    is_checked = True
            else:
                feedback = 'The previous command is not valid. Use ```Edit``` to edit the code'
            
            if is_checked is True:
                self.statistics_per_phase["data acquisition"]["steps"] = _i
                self.set_agent_attr("prev_code", new_code)
                self.set_agent_attr("data_summary", code_resp)
                self.reset_agents()
                return False

    def paper_lineage(self):
        arx_eng = ArxivSearch()
        max_tries = self.max_steps
        summary = self.researchagent.paper[0]['summary']
        full_text = self.researchagent.paper[0]['full_text']
        for _i in range(1, max_tries):
            feedback_ = f"Previous generated summary of the paper {summary}"
            info_ = f"The full text of the paper that need to replicate: {full_text}"
            resp = self.researchagent.inference(self.instruction, "paper lineage", feedback=feedback_, info=info_, step=_i, temp=0.8)
            if "```ADD" in resp:
                related_list = extract_prompt(resp, "ADD")
                print("ADD\n", resp)
                related_list = ast.literal_eval(related_list)

                if not isinstance(related_list, list):
                    print("The related list should be the format of list")
                    info_ += 'The format is incorrect. You need to provide a list to return the relevant papers.'
                    continue
                for i_, related_ in enumerate(related_list):
                    # TBD
                    # if related_ == TITLE:
                    #     print("The related work is the same as the paper title. You should not add it.")
                    #     continue
                    related_ = related_.replace("-", " ")
                    print(f"The related work {i_+1}: {related_}")
                    related_id = arx_eng.get_arxiv_id_by_title(related_)
                    if related_id is None:
                        print(f"The related work {related_} is not found in arxiv.")
                    arxiv_paper = arx_eng.retrieve_full_paper_text(related_id)
                    with open(f"{self.related_work}", "r", encoding="utf-8") as file:
                        arxiv_paper = file.read()
                    if arxiv_paper != "EXTRACTION FAILED":
                        info_ = arxiv_paper + "\nThe above is a related work of the paper that you need to replicate. Please review method and experiment proposed the paper and extract github link with the format [method](github link)."
                        review_content = self.researchagent.inference(self.instruction, "overall summary", info=info_, step=_i, temp=0.8, use_command=False)
                        corr_link = extract_markdown_links(review_content)
                        print('Extracted GitHub link', corr_link)
                        if len(corr_link) == 0:
                            pass
                        else:
                            file_path = os.path.join('paper_lineage', f'{related_}.json')
                            if os.path.exists(file_path):
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    repo_code = json.load(f)
                            else:
                                print('Downloadin from Github. This might take a few minutes. ...')
                                repo_code = get_repo_structure(corr_link[0][1])
                                save_results(repo_code, output_dir='code_repo', index=related_)
                            self.researchagent.add_review(arxiv_paper, review_content)
                            repo_code = convert_json_to_string(repo_code)

                            success_select_file = False
                            while not success_select_file:
                                info_ = f"The comparison of related paper with paper you are replicate: {review_content}."
                                feedback_ = f"{repo_code}\nThe code repository mentioned above is related to the paper that needs to be replicated."
                                feedback_ += "Please filter out the code files that are helpful for the experiments, and return the useful code files in the form of ```python ['file1.py', 'model/file2.py']```."
                                resp = self.codeagent.inference(self.instruction, "paper lineage", feedback=feedback_, info=info_, step=_i)
                                if "```ADD" in resp:
                                    file_list = extract_prompt(resp, "ADD")
                                    print('ADD\n', file_list)
                                    file_list = ast.literal_eval(file_list)
                                    if not isinstance(file_list, list):
                                        print("The returned file_list needs to be a list.")
                                        info_ += 'The format is incorrect. You need to provide a list to return the relevant papers.'
                                        continue
                                    related_code = extract_files_by_paths(file_path, file_list)
                                    code_content = ''
                                    for path, content in related_code.items():
                                        code_content += f'########\nFile:{path}\n\nContent:\n{content}\n'
                                    self.codeagent.add_code(code_content, review_content)
                                    success_select_file = True
                return False

    def overall_summary(self):
        max_tries = self.max_steps
        with open(f"{self.paper_path}", "r", encoding="utf-8") as file:
            arxiv_paper = file.read()
        review_content = str()
        for _i in range(1, max_tries):
            info_ = f'The full text that need to replicate: {arxiv_paper}'
            review_content = self.researchagent.inference(self.instruction, "overall summary", info=info_, step=_i, temp=0.8, use_command=False)
            print("REVIEW\n", review_content)
            self.researchagent.add_review(arxiv_paper, review_content)
            self.set_agent_attr("paper", self.researchagent.paper)
            self.reset_agents()
            self.statistics_per_phase["overall summary"]["steps"] = _i
            return False

    def method_summary(self, use_image=False):
        max_tries = self.max_steps
        with open(f"{self.paper_path}", "r", encoding="utf-8") as file:
            arxiv_paper = file.read()
        for _i in range(1, max_tries):
            info_ = f'The full text that need to replicate: {arxiv_paper}. For each innovation point in the method summary, it would be best to have a formula to assist in understanding. Do not need to generate code here, just review.'
            review_content = self.researchagent.inference(self.instruction, "method summary", info=info_, step=_i, temp=0.8, use_command=False)
            print("METHOD SUMMARY\n", review_content)
            if use_image == True:
                pass
                # TBD
            self.researchagent.update_review(review_content, section='method')
            self.reset_agents()
            self.statistics_per_phase["method summary"]["steps"] = _i
            return False

    def experiment_summary(self, use_image=False):
        max_tries = self.max_steps
        with open(f"{self.paper_path}", "r", encoding="utf-8") as file:
            arxiv_paper = file.read() 
        for _i in range(max_tries):
            info_ = f'The full text that need to replicate: {arxiv_paper}. Do not need to generate code here, just review.'
            experiment_content = self.researchagent.inference(self.instruction, "experiment summary", info=info_, step=_i, temp=0.8, use_command=False)
            print("EXPERIMENT REVIEW", experiment_content)
            if use_image == True:
                pass
                # TBD
            self.researchagent.update_review(experiment_content, section='experiment')
            self.reset_agents()
            self.statistics_per_phase["experiment summary"]["steps"] = _i
            
            return False


def parse_arguments():
    parser = argparse.ArgumentParser(description="AutoReproduce Workflow")

    parser.add_argument( '--deepseek-api-key', type=str,help='Provide the DeepSeek API key.')
    parser.add_argument('--arxiv-id', type=str, default='', help='Specify the arxiv id.')
    parser.add_argument('--api-key', type=str,default="", help='Provide the OpenAI API key.')
    parser.add_argument('--llm-backend', type=str, default="o3-mini", help='Backend LLM to use for agents in Agent Laboratory.')
    parser.add_argument('--paper-path', type=str, default="examples/dkd/paper.txt", help='Paper Txt File Path')
    parser.add_argument('--dataloader-path', type=str, default="examples/dkd/dataloader.py", help='Paper Path')
    parser.add_argument('--import-exp-config', type=bool, default=True, help='Utilize config in evaluation/exp_config.py')
    parser.add_argument('--instructions', type=str, default="", help='Instruction about the experiment to reproduce.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    llm_backend = args.llm_backend
    api_key = os.getenv('OPENAI_API_KEY') or args.api_key
    deepseek_api_key = os.getenv('DEEPSEEK_API_KEY') or args.deepseek_api_key
    if args.api_key is not None and os.getenv('OPENAI_API_KEY') is None:
        os.environ["OPENAI_API_KEY"] = args.api_key
    if args.deepseek_api_key is not None and os.getenv('DEEPSEEK_API_KEY') is None:
        os.environ["DEEPSEEK_API_KEY"] = args.deepseek_api_key

    task_notes_LLM = [
        {"phases": ["overall summary"],
         "note": "The summary you generate will serve as a guide for the subsequent steps in the paper replication process. Therefore, your summary needs to include all the details relevant to the paper. In order to better summarize the algorithms in the paper, you need to emphasize all the formulas and algorithms presented in the article in your summary. Your reseponse must in English."},

        {"phases": ["method replication", "experiment execution"],
         "note": f"If you want to use the content from the related repo, you need to generate the complete code instead of just importing. Do not add or edit the summary of the paper at the beginning of the paper." },

        {"phases": ["experiment execution"],
         "note": "You need to reproduce the entire experimental process, so the code should include both the training validation and evaluation phases. The training loss and evaluation performance must be reported. In order to prevent the situation where one doesn't know what needs to be modified for debugging, you can add a print function for relevant data before the line where the bug occurs."},

        {"phases": ["method replication"],
         "note": "You need to use the data in the existing dataloader to check whether the dimensions of the model's input and output conform to the requirements. In order to assist find data shape error, you can use ```print()``` function before the error line to check the shape and type of the data that reproduces the error. Do not only generate model class code, you need to execute the model code using the data to check whether the model code is correct."},

        {"phases": ["data acquisition"],
         "note": "Do not print the data themselves, such as input and output data, the data value is too many. The data shape, dtype and type are the much more important."},

        {"phases": ["experiment execution"],
         "note": "Make sure the model should be trained on GPUs. The training and testing information should be print at the end of each epoch."},
    ]

    paper_path = args.paper_path
    if args.dataloader_path is not None:
        with open(f"{args.dataloader_path}", "r", encoding="utf-8") as file:
            dataloader_code = file.read()
    else:
        dataloader_code = None

    if args.import_exp_config is True:
        arxiv_id = ARXIV_ID
        task = TASK
        title = TITLE
        instruction = INSTRUCTION
    else:
        ## TBD need to define
        arxiv_id = ''
        task = ''
        title = ''
        instruction = args.instructions

    lab = AutoReproduce(
        arxiv_id=arxiv_id,
        paper_path=paper_path,
        instruction=instruction,
        dataloader_code=dataloader_code,
        notes=task_notes_LLM,
        agent_model_backbone=llm_backend,
    )

    lab.perform_research()






