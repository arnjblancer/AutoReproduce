from utils import *
import requests
import time
import arxiv
import os, re
import io, sys
import concurrent.futures
from semanticscholar import SemanticScholar
import json
import traceback
import concurrent.futures
import fitz
import subprocess
import tempfile
import matplotlib
import ast
from pathlib import Path

def is_valid_pdf(pdf_path):
    try:
        with open(pdf_path, "rb") as f:
            return f.read(4) == b'%PDF'
    except:
        return False
    

class SemanticScholarSearch:
    def __init__(self):
        self.sch_engine = SemanticScholar(retry=False)

    def find_papers_by_str(self, query, N=10):
        paper_sums = list()
        results = self.sch_engine.search_paper(query, limit=N, min_citation_count=3, open_access_pdf=True)
        for _i in range(len(results)):
            paper_sum = f'Title: {results[_i].title}\n'
            paper_sum += f'Abstract: {results[_i].abstract}\n'
            paper_sum += f'Citations: {results[_i].citationCount}\n'
            paper_sum += f'Release Date: year {results[_i].publicationDate.year}, month {results[_i].publicationDate.month}, day {results[_i].publicationDate.day}\n'
            paper_sum += f'Venue: {results[_i].venue}\n'
            paper_sum += f'Paper ID: {results[_i].externalIds["DOI"]}\n'
            paper_sums.append(paper_sum)
        return paper_sums

    def retrieve_full_paper_text(self, query):
        pass

    def get_paper_references(self, query):
        pass


class ArxivSearch:
    def __init__(self):
        # Construct the default API client.
        self.sch_engine = arxiv.Client()
        
    def _process_query(self, query: str) -> str:
        """Process query string to fit within MAX_QUERY_LENGTH while preserving as much information as possible"""
        MAX_QUERY_LENGTH = 300
        
        if len(query) <= MAX_QUERY_LENGTH:
            return query
        
        # Split into words
        words = query.split()
        processed_query = []
        current_length = 0
        
        # Add words while staying under the limit
        # Account for spaces between words
        for word in words:
            # +1 for the space that will be added between words
            if current_length + len(word) + 1 <= MAX_QUERY_LENGTH:
                processed_query.append(word)
                current_length += len(word) + 1
            else:
                break
            
        return ' '.join(processed_query)
    
    def find_papers_by_str(self, query, N=20):
        processed_query = self._process_query(query)
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                search = arxiv.Search(
                    query="abs:" + processed_query,
                    max_results=N,
                    sort_by=arxiv.SortCriterion.Relevance)

                paper_sums = list()
                # `results` is a generator; you can iterate over its elements one by one...
                for r in self.sch_engine.results(search):
                    paperid = r.pdf_url.split("/")[-1]
                    pubdate = str(r.published).split(" ")[0]
                    paper_sum = f"Title: {r.title}\n"
                    paper_sum += f"Summary: {r.summary}\n"
                    paper_sum += f"Publication Date: {pubdate}\n"
                    paper_sum += f"Categories: {' '.join(r.categories)}\n"
                    paper_sum += f"arXiv paper ID: {paperid}\n"
                    paper_sums.append(paper_sum)
                time.sleep(2.0)
                return "\n".join(paper_sums)
                
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    # 递增延时
                    time.sleep(2 * retry_count)
                    continue
                
        return None

    def get_arxiv_id_by_title(self, paper_title):
        client = arxiv.Client()  # 显式创建Client实例
        search = arxiv.Search(
            query=f'ti:"{paper_title}"',
            max_results=1,
            sort_by=arxiv.SortCriterion.Relevance
        )
        try:
            result = next(client.results(search))
            return result.entry_id.split('/')[-1]  # 返回arXiv ID（如"1706.03762v7"）
        except (StopIteration, Exception) as e:
            print(f"搜索失败: {e}")
            return None

    def retrieve_full_paper_text(self, query):
        pdf_text = str()
        paper = next(arxiv.Client().results(arxiv.Search(id_list=[query])))
        # Download the PDF to the PWD with a custom filename.
        paper.download_pdf(filename="downloaded-paper.pdf")
        if not os.path.exists("downloaded-paper.pdf"):
            return "DOWNLOAD FAILED" 
        # creating a pdf reader object
        doc = fitz.open('downloaded-paper.pdf')
        # Iterate over all the pages
        for page_number in range(len(doc)):
            # Extract text from the page
            page = doc.load_page(page_number)
            try:
                text = page.get_text("text")
            except Exception as e:
                print(f"Extraction failed: {str(e)}")
                os.remove("downloaded-paper.pdf")
                time.sleep(2.0)
                return "EXTRACTION FAILED"

            # Do something with the text (e.g., print it)
            pdf_text += f"--- Page {page_number} ---"
            pdf_text += text
            pdf_text += "\n"
        doc.close()
        os.remove("downloaded-paper.pdf")
        time.sleep(2.0)

        return pdf_text

import io
import sys
import traceback
import concurrent.futures

import io
import sys
import traceback
import multiprocessing
import io
import sys
import traceback


def execute_code(code_str, timeout=120, MAX_LEN=100000):
    # Prevent plotting errors
    matplotlib.use('Agg')

    # Safety checks
    if "load_dataset('pubmed" in code_str:
        return None, "[CODE EXECUTION ERROR] pubmed Download took way too long. Program terminated"
    if "exit(" in code_str:
        return None, "[CODE EXECUTION ERROR] The exit() command is not allowed you must remove this."

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
        temp_file.write(code_str)
        temp_file_path = temp_file.name

    try:
        result = subprocess.run(
            ["python", temp_file_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        output = result.stdout[-MAX_LEN:]
        error_output = result.stderr[-MAX_LEN:] if result.stderr else ""

        # 通用警告检测模式
        warning_pattern = r"^(.*?):\d+:\s*\w*Warning:"
        is_warning_only = (
            re.search(warning_pattern, error_output, re.MULTILINE) and 
            not re.search(r"Error:|Exception:|Traceback", error_output, re.MULTILINE) and
            result.returncode == 0
        )

        # 如果是纯警告情况
        if is_warning_only:
            return output, None
        
        # 其他情况按原逻辑处理
        if result.returncode != 0 or error_output:
            return output, f"[CODE EXECUTION ERROR]: {error_output or f'Process exited with code {result.returncode}'}"[:MAX_LEN]
        
        return output, None

    except subprocess.TimeoutExpired as e:
        return None, f"[CODE EXECUTION ERROR]: Timeout after {timeout} seconds"[:MAX_LEN]
    except Exception as e:
        return None, f"[CODE EXECUTION ERROR]: {str(e)}"[:MAX_LEN]
    finally:
        sys.stdout = sys.__stdout__
        try:
            os.unlink(temp_file_path)
        except OSError:
            pass

def delete_with_extensions(json_data):
    if isinstance(json_data, dict):
        keys_to_delete = [
            key for key in json_data.keys()
            if re.search(r'\.(png|jpg|pdf)$', key, re.IGNORECASE)
        ]
        for key in keys_to_delete:
            del json_data[key]
        for value in json_data.values():
            if isinstance(value, (dict, list)):
                delete_with_extensions(value)
    elif isinstance(json_data, list):
        for item in json_data:
            if isinstance(item, (dict, list)):
                delete_with_extensions(item)

def extract_markdown_links(text):
    pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    return re.findall(pattern, text)

def get_repo_structure(repo_url, token=None):
    """
    {
        "repo_name": "repo-name",
        "files": [
            {
                "path": "full/file/path",
                "content": "file content"
            },
            ...
        ]
    }
    """
    parts = repo_url.rstrip('/').split('/')
    username = parts[-2]
    repo_name = parts[-1]
    
    api_base_url = f"https://api.github.com/repos/{username}/{repo_name}/contents"
    
    headers = {'Authorization': f'token {token}'} if token else {}
    
    def fetch_contents(url):
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"{response.status_code}")
            if response.status_code == 403:
                error_msg = response.json().get("message", "")
                print(error_msg)
            return []
        return response.json()
    
    def get_file_content(download_url):
        response = requests.get(download_url, headers=headers)
        return response.text if response.status_code == 200 else None
    
    def traverse_directory(url, current_path=""):
        contents = fetch_contents(url)
        files_data = []
        
        for item in contents:
            item_path = f"{current_path}/{item['name']}" if current_path else item['name']
            
            if item['type'] == 'file':
                if item['name'].lower().endswith(('.py', '.sh', '.md')):
                    content = get_file_content(item['download_url'])
                    files_data.append({
                        "path": item_path,
                        "content": content if content else "[None]"
                    })
                    
            elif item['type'] == 'dir':
                files_data.extend(traverse_directory(item['url'], item_path))
        
        return files_data

    all_files = traverse_directory(api_base_url)
    
    return {
        "repo_name": repo_name,
        "files": all_files
    }

def save_results(result, output_dir='code_repo', index=1):
    os.makedirs(output_dir, exist_ok=True)
    repo_path = os.path.join(output_dir, f'repo_content_{index}.json')
    with open(repo_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

def parse_directory_structure(tree_str):
    lines = tree_str.strip().split('\n')
    root = {}
    stack = [(root, -1)]

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("Repo"):
            continue

        indent = len(line) - len(line.lstrip())
        line_content = stripped.split('|-- ')[-1]

        while stack and indent <= stack[-1][1]:
            stack.pop()

        if not stack:
            continue

        current_dict, _ = stack[-1]
        if '.' in line_content:
            current_dict[line_content] = ""
        else: 
            current_dict[line_content] = {}
            stack.append((current_dict[line_content], indent))

    return root

def extract_matching_keys(tree_dict, file_content):
    result = {}

    if isinstance(file_content, dict):
        if len(file_content) == 1 and isinstance(list(file_content.values())[0], dict):
            file_content = list(file_content.values())[0]
        
        for key, value in file_content.items():
            if key in tree_dict:
                if isinstance(value, dict) and isinstance(tree_dict[key], dict):
                    sub_result = extract_matching_keys(tree_dict[key], value)
                    if sub_result:
                        result[key] = sub_result
                else:
                    result[key] = value
    return result

def generate_code_lines(code, start=0, align=2):
    """
    Generate well-formatted code lines with line numbers
    @param code: (list) list of code line strings
    @param start: (int) starting line number (default 0)
    @param align: (int) number of digits to align (default 4)
    @return: (str) code lines formatted with line numbers
    """

    code_list = code.split("\n") 
    if not code_list:
        return ""
    max_length = len(str(start + len(code_list) - 1))
    align = max(align, max_length)
    return code_list, "\n".join(f"{i + start:>{align}}|{line}" 
                    for i, line in enumerate(code_list))

def edit_code_line(begin, end, code_list, fix_code):
    current_code = code_list.copy() 
    lines_to_add = list(reversed(fix_code))

    end = min(end, len(current_code) - 1)
    lines_to_replace = list(reversed(range(begin, end + 1)))
    
    for _ln in lines_to_replace:
        if _ln < len(current_code):
            current_code.pop(_ln)
    
    for _line in lines_to_add:
        current_code.insert(begin, _line)
    
    return "\n".join(current_code)
    
def delete_code_line(begin, end, code_list):
    try:
        current_code = code_list
        lines_to_replace = list(reversed(range(begin, end+1)))
        for _ln in lines_to_replace:
            current_code.pop(_ln)
        new_code = "\n".join(current_code)
        return new_code
    except Exception as e:
        return None

def get_content_with_number_start(code_list):
    pattern = r'^\s*\d+.*'
    match_indices = []
    
    for i, line in enumerate(code_list):
        if re.match(pattern, line):
            match_indices.append(i)

    result = []
    for i in range(len(match_indices)):
        start = match_indices[i]
        if i < len(match_indices) - 1:
            end = match_indices[i + 1]
            result.append(code_list[start:end])
        else:
            result.append(code_list[start:])
    
    return result

def extract_file_list(text):
    """Extract the file list from a markdown code block and return as Python list"""
    # Pattern to match Python code blocks with list content
    pattern = r'```python\s*\[(.*?)\]```'
    
    # Find all matches (using re.DOTALL to match across newlines)
    matches = re.search(pattern, text, re.DOTALL)
    
    if matches:
        list_content = matches.group(1).strip()
        
        # Clean up the content by:
        # 1. Removing line comments
        # 2. Keeping only lines that look like list items
        cleaned_lines = []
        for line in list_content.split('\n'):
            line = line.strip()
            # Remove Python comments
            line = re.sub(r'#.*$', '', line).strip()
            if line:
                # Ensure proper quoting if not already present
                if (not line.startswith('"') and not line.startswith("'") and 
                    not line.endswith('"') and not line.endswith("'")):
                    line = f"'{line}'"
                cleaned_lines.append(line)
        
        # Reconstruct as a proper list string
        list_str = '[' + ', '.join(cleaned_lines) + ']'
        
        # Safely evaluate using ast.literal_eval
        try:
            return ast.literal_eval(list_str)
        except (SyntaxError, ValueError):
            pass

def extract_files_by_paths(json_file_path, target_paths):
    if not Path(json_file_path).exists():
        print(f"Error: file {json_file_path} does not exist")
        return {}
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        return {}
    except Exception as e:
        print(f"File reading error: {e}")
        return {}
    
    result = {}
    target_paths = set(target_paths)
    
    if "files" in data and isinstance(data["files"], list):
        for file_info in data["files"]:
            if isinstance(file_info, dict) and "path" in file_info:
                path = file_info["path"]
                if path in target_paths:
                    result[path] = file_info.get("content", "[Content not exist]")
    
    missing_paths = target_paths - set(result.keys())
    if missing_paths:
        print(f"Warning: The following paths were not found in the JSON: {missing_paths}")
    
    return result

def convert_json_to_string(json_data):
    result = []
    for file in json_data["files"]:
        result.append(f"########\nFile: {file['path']}\nContent: {file['content']}\n")
    return "".join(result)