import time, tiktoken
from openai import OpenAI
import openai
import os, anthropic, json
import base64
from pdf2image import convert_from_path
import io
TOKENS_IN = dict()
TOKENS_OUT = dict()

encoding = tiktoken.get_encoding("o200k_base")

def curr_cost_est():
    costmap_in = {
        "gpt-4o": 2.50 / 1000000,
        "gpt-4o-mini": 0.150 / 1000000,
        "o1-mini": 1.1 / 1000000,
        "claude-3-5-sonnet": 3.00 / 1000000,
        "deepseek-chat": 0.3 / 1000000,
        "deepseek-r1": 0.6 / 1000000,
        "o1": 15.00 / 1000000,
        "o3-mini": 0.25 / 1000000,
        "gpt-5": 5.00 / 1000000,
    }
    costmap_out = {
        "gpt-4o": 10.00/ 1000000,
        "gpt-4o-mini": 0.6 / 1000000,
        "o1-mini": 4.4 / 1000000,
        "claude-3-5-sonnet": 12.00 / 1000000,
        "deepseek-chat": 1.1 / 1000000,
        "deepseek-r1": 2.2 / 1000000,
        "o1": 60.00 / 1000000,
        "o3-mini": 1.25 / 1000000,
        "gpt-5": 15.00 / 1000000
    }
    return sum([costmap_in[_]*TOKENS_IN[_] for _ in TOKENS_IN]) + sum([costmap_out[_]*TOKENS_OUT[_] for _ in TOKENS_OUT])

def query_model(model_str, prompt, system_prompt='', tries=5, timeout=5.0, temp=None, print_cost=True, version="1.5"):
    if system_prompt == '':
        system_prompt = "You are a helpful assistant."
    openai_api_key = os.getenv('OPENAI_API_KEY')
    base_url = os.getenv('BASE_URL', 'https://api.openai.com/v1')
    if openai_api_key is not None:
        openai.api_key = openai_api_key
    for _ in range(tries):
        if model_str == "gpt-4o-mini" or model_str == "gpt4omini" or model_str == "gpt-4omini" or model_str == "gpt4o-mini":
            model_str = "gpt-4o-mini"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}]
            if version == "0.28":
                if temp is None:
                    completion = openai.ChatCompletion.create(
                        model=f"{model_str}",  # engine = "deployment_name".
                        messages=messages
                    )
                else:
                    completion = openai.ChatCompletion.create(
                        model=f"{model_str}",  # engine = "deployment_name".
                        messages=messages, temperature=temp
                    )
            else:
                client = OpenAI(api_key=openai_api_key, base_url=base_url)
                if temp is None:
                    completion = client.chat.completions.create(
                        model="gpt-4o-mini", messages=messages)
                else:
                    completion = client.chat.completions.create(
                        model="gpt-4o-mini", messages=messages, temperature=temp)
            answer = completion.choices[0].message.content
        elif model_str == "gpt-5" or model_str == "gpt5":
            model_str = "gpt-5"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}]
            client = OpenAI(api_key=openai_api_key, base_url=base_url)
            if temp is None:
                completion = client.chat.completions.create(
                    model="gpt-5", messages=messages)
            else:
                completion = client.chat.completions.create(
                    model="gpt-5", messages=messages, temperature=temp)
            answer = completion.choices[0].message.content
        elif model_str == "claude-3.5-sonnet":
                client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
                message = client.messages.create(
                    model="claude-3-5-sonnet-latest",
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}])
                answer = json.loads(message.to_json())["content"][0]["text"]
        elif model_str == "deepseek-r1":
            model_str = "deepseek-r1"

            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}]

            client = OpenAI(
                api_key=os.getenv('DEEPSEEK_API_KEY'),
                base_url="https://api.deepseek.com/v1"
            )
            if temp is None:
                completion = client.chat.completions.create(
                    model="DeepSeek-R1", messages=messages)
            else:
                completion = client.chat.completions.create(
                    model="DeepSeek-R1", messages=messages, temperature=temp)
            answer = completion.choices[0].message.content

        elif model_str == "gpt4o" or model_str == "gpt-4o":
            model_str = "gpt-4o"
            
            messages = [
                {"role": "system", "content": system_prompt}
            ]

            user_content = []
            
            if isinstance(prompt, tuple) and len(prompt) == 2:
                text, image = prompt
                
                if text:
                    user_content.append({"type": "text", "text": text})

                if image:
                    image_ext = os.path.splitext(image)[1].lower()
                    if image_ext == ".pdf":
                        img = convert_from_path(image, dpi=300, fmt="png")[0]
                        # PIL Image->Base64
                        buffered = io.BytesIO()
                        img.save(buffered, format="PNG")
                        image_url = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    else:
                        with open(image, "rb") as f:
                            image_url = base64.b64encode(f.read()).decode("utf-8")
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_url}"}
                    })
            else:
                user_content.append({"type": "text", "text": prompt})
            messages.append({"role": "user", "content": user_content})

            if version == "0.28":
                if temp is None:
                    completion = openai.ChatCompletion.create(
                        model=f"{model_str}",  # engine = "deployment_name".
                        messages=messages
                    )
                else:
                    completion = openai.ChatCompletion.create(
                        model=f"{model_str}",  # engine = "deployment_name".
                        messages=messages, temperature=temp)
            else:
                client = OpenAI(
                    api_key=openai_api_key,
                    base_url=base_url
                )
                if temp is None:
                    completion = client.chat.completions.create(
                        model="gpt-4o-2024-08-06", messages=messages)
                else:
                    completion = client.chat.completions.create(
                        model="gpt-4o-2024-08-06", messages=messages, temperature=temp)
            answer = completion.choices[0].message.content
        elif model_str == "deepseek-chat":
            model_str = "deepseek-chat"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}]
            if version == "0.28":
                raise Exception("Please upgrade your OpenAI version to use DeepSeek client")
            else:
                deepseek_client = OpenAI(
                    api_key=os.getenv('DEEPSEEK_API_KEY'),
                    base_url="https://api.deepseek.com/v1"
                )
                if temp is None:
                    completion = deepseek_client.chat.completions.create(
                        model="deepseek-chat",
                        messages=messages)
                else:
                    completion = deepseek_client.chat.completions.create(
                        model="deepseek-chat",
                        messages=messages,
                        temperature=temp)
            answer = completion.choices[0].message.content
        elif model_str == "o1-mini":
            model_str = "o1-mini"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}]
            if version == "0.28":
                completion = openai.ChatCompletion.create(
                    model=f"{model_str}",  # engine = "deployment_name".
                    messages=messages)
            else:
                client = OpenAI(
                    api_key=openai_api_key,
                    base_url=base_url
                )
                completion = client.chat.completions.create(
                    model="o1-mini", messages=messages)
            answer = completion.choices[0].message.content
        elif model_str == "o3-mini":
            model_str = "o3-mini"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}]
            if version == "0.28":
                completion = openai.ChatCompletion.create(
                    model=f"{model_str}",  # engine = "deployment_name".
                    messages=messages)
            else:
                client = OpenAI(
                    api_key=openai_api_key,
                    base_url=base_url
                )
                completion = client.chat.completions.create(
                    model="o3-mini", messages=messages)
            answer = completion.choices[0].message.content
        elif model_str == "o1":
            model_str = "o1"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}]
            if version == "0.28":
                completion = openai.ChatCompletion.create(
                    model="o1-2024-12-17",  # engine = "deployment_name".
                    messages=messages)
            else:
                client = OpenAI(
                    api_key=openai_api_key,
                    base_url=base_url
                )
                completion = client.chat.completions.create(
                    model="o1-2024-12-17", messages=messages)
        if model_str in ["o1-mini", "claude-3-5-sonnet", "o3-mini"]:
            encoding = tiktoken.encoding_for_model("gpt-4o")
        elif model_str in ["deepseek-chat", "deepseek-r1"]:
            encoding = tiktoken.encoding_for_model("cl100k_base")
        elif model_str == "gpt-5":
            encoding = tiktoken.get_encoding("o200k_base")
        else:
            encoding = tiktoken.encoding_for_model(model_str)
        if model_str not in TOKENS_IN:
            TOKENS_IN[model_str] = 0
            TOKENS_OUT[model_str] = 0
        if isinstance(prompt, tuple) and len(prompt) == 2:
            TOKENS_IN[model_str] += len(encoding.encode(system_prompt + prompt[0]))
        else:
            TOKENS_IN[model_str] += len(encoding.encode(system_prompt + prompt))
        TOKENS_OUT[model_str] += len(encoding.encode(answer))
        if print_cost:
            print(f"Current experiment cost = ${curr_cost_est()}, ** Approximate values, may not reflect true cost")
        return answer
    raise Exception("Max retries: timeout")

#print(query_model(model_str="o1-mini", prompt="hi", system_prompt="hey"))