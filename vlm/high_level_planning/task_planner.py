import openai
import tiktoken
import json
import os
import re
import time

enc = tiktoken.get_encoding("cl100k_base")
dir_system = os.path.join(os.path.dirname(__file__), 'system')
dir_prompt = os.path.join(os.path.dirname(__file__), 'prompt')
dir_query = os.path.join(os.path.dirname(__file__), 'query')

prompt_load_order = ['role_definition',
                     'environment_format',
                     'function_list',
                     'output_format',
                     'examples']


class planner:
    def __init__(self, use_azure=True):
        self.use_azure = use_azure
        if self.use_azure:
            self.api_version = "2023-09-01-preview"
            self.client = openai.AzureOpenAI(
                # https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#rest-api-versioning
                api_version=self.api_version,
                # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
                azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
                api_key=os.environ.get("AZURE_OPENAI_API_KEY")
            )
        else:
            self.client = openai.OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY")
            )
        self.messages = []
        self.max_token_length = 8000
        self.max_completion_length = 500
        self.last_response = None
        self.last_response_raw = None
        self.query = ''
        self.instruction = ''
        self.current_time = time.time()
        self.waittime_sec = 5
        self.time_api_called = time.time() - self.waittime_sec
        self.retry_count_tolerance = 10
        # load prompt file
        fp_system = os.path.join(dir_system, 'system.txt')
        with open(fp_system) as f:
            data = f.read()
        self.system_message = {"role": "system", "content": data}

        for prompt_name in prompt_load_order:
            fp_prompt = os.path.join(dir_prompt, prompt_name + '.txt')
            with open(fp_prompt) as f:
                data = f.read()
            data_spilit = re.split(r'\[user\]\n|\[assistant\]\n', data)
            data_spilit = [item for item in data_spilit if len(item) != 0]
            # it start with user and ends with system
            assert len(data_spilit) % 2 == 0
            for i, item in enumerate(data_spilit):
                if i % 2 == 0:
                    self.messages.append({"sender": "user", "text": item})
                else:
                    self.messages.append({"sender": "assistant", "text": item})
        fp_query = os.path.join(dir_query, 'query.txt')
        with open(fp_query) as f:
            self.query = f.read()

    def reset_history(self):  # clear the conversation history and reset the prompt
        self.messages = []
        for prompt_name in prompt_load_order:
            fp_prompt = os.path.join(dir_prompt, prompt_name + '.txt')
            with open(fp_prompt) as f:
                data = f.read()
            data_spilit = re.split(r'\[user\]\n|\[assistant\]\n', data)
            data_spilit = [item for item in data_spilit if len(item) != 0]
            # it start with user and ends with system
            assert len(data_spilit) % 2 == 0
            for i, item in enumerate(data_spilit):
                if i % 2 == 0:
                    self.messages.append({"sender": "user", "text": item})
            else:
                self.messages.append({"sender": "assistant", "text": item})

    def create_prompt(self):
        prompt = []
        prompt.append(self.system_message)
        for message in self.messages:
            prompt.append(
                {"role": message['sender'], "content": message['text']})
        prompt_content = ""
        for message in prompt:
            prompt_content += message["content"]
        # print('prompt length: ' + str(len(enc.encode(prompt_content))))
        if len(enc.encode(prompt_content)) > self.max_token_length - \
                self.max_completion_length:
            print('prompt too long. truncated.')
            # truncate the prompt by removing the oldest two messages
            self.messages = self.messages[2:]
            prompt = self.create_prompt()
        return prompt

    def extract_json_part(self, text):
        # JSON part is between ```python and ``` on a new line
        try:
            start = text.index('```python') + len('```python')
            end = text.index('```', start)
            # Removing any leading/trailing whitespace
            text_json = text[start:end].strip()
            text_json = text_json.replace("'", "\"")
            return text_json
        except ValueError:
            # This means '```python' or closing '```' was not found in the text
            return None

    def generate(self, message, environment, is_user_feedback=False):
        if is_user_feedback:
            self.messages.append({'sender': 'user',
                                  'text': message})
        else:
            text_base = self.query
            if text_base.find('[ENVIRONMENT]') != -1:
                text_base = text_base.replace(
                    '[ENVIRONMENT]', json.dumps(environment))
            if text_base.find('[INSTRUCTION]') != -1:
                text_base = text_base.replace('[INSTRUCTION]', message)
                self.instruction = text_base
            self.messages.append({'sender': 'user', 'text': text_base})

        self.current_time = time.time()
        time_diff = self.current_time - self.time_api_called
        if time_diff < self.waittime_sec:
            print("waiting for " + str(self.waittime_sec - time_diff) + " seconds...")
            time.sleep(self.waittime_sec - time_diff)
        retry_count = 0
        while True:
            try:
                if self.use_azure:
                    deployment_name = os.environ.get(
                        "AZURE_OPENAI_DEPLOYMENT_NAME")
                    response = self.client.chat.completions.create(
                        model=deployment_name,
                        messages=self.create_prompt(),
                        temperature=2.0,
                        max_tokens=self.max_completion_length,
                        top_p=0.5,
                        frequency_penalty=0.0,
                        presence_penalty=0.0)
                    text = response.choices[0].message.content
                else:
                    response = self.client.chat.completions.create(
                        model="gpt-4",
                        messages=self.create_prompt(),
                        temperature=2.0,
                        max_tokens=self.max_completion_length,
                        top_p=0.5,
                        frequency_penalty=0.0,
                        presence_penalty=0.0)
                    text = response.choices[0].message.content
                self.time_api_called = time.time()
                try:
                    # analyze the response
                    tmp_last_response = text
                    tmp_last_response = self.extract_json_part(
                        tmp_last_response)
                    self.json_dict = json.loads(
                        tmp_last_response, strict=False)
                    break
                except BaseException:
                    print("api call failed. retrying...")
                    retry_count += 1
                    if retry_count > self.retry_count_tolerance:
                        raise Exception("api call failed")
                    continue
            except Exception as e:
                print(e)
                match = re.search("retry after (\\d+) seconds", e.args[0])
                wait_time = int(match.group(1))
                print(
                    "api call failed due to rate limit. waiting for " +
                    str(wait_time) +
                    " seconds...")
                time.sleep(wait_time)
                continue
        self.last_response = tmp_last_response
        self.last_response_raw = text
        self.environment = self.json_dict["environment_after"]
        self.messages.append(
            {"sender": "assistant", "text": self.last_response_raw})
        return self.json_dict
