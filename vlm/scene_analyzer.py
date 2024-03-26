from openai import OpenAI, AzureOpenAI
import os
import json


def scene_understanding(frame, instruction, use_azure=True):
    def extract_json_part(text):
        try:
            start = text.index('```python') + len('```python')
            end = text.index('```', start)
            text_json = text[start:end].strip()
            text_json = text_json.replace("'", "\'")
            return text_json
        except ValueError:
            return None
    PROMPT_MESSAGE = """This is a scene in which a human is doing "[ACTION]". Understand this scene and generate a scenery description to assist in task planning:
                Information about environments is given as python dictionary. For example:
                {
                    "objects": [
                        "<cup>",
                        "<office_table>"],
                    "object_properties": {
                        "<cup>": ["GRABBABLE"],
                        "<office_table>":[]},
                    "spatial_relations": {
                        "<cup>": ["on(<office_table>)"],
                        "<office_table>":[]},
                    "your_explanation": "Human is picking up the cup from the office table and placing it back on the table. I omitted the juice on the table as it is not being manipulated."
                }
                - The "objects" field denotes the list of objects. Enclose the object names with '<' and '>'. Connect the words without spaces, using underscores instead. Do not include human beings in the object list.
                - The "object_properties" field denotes the properties of the objects. Objects have the following properties:
                - GRABBABLE: If an object has this attribute, it can be potentially grabbed by the robot.
                - The "spatial_relations" field denotes the list of relationships between objects. Use only the following functions to describe these relations: [inside(), on()]. For example, 'on(<office_table>)' indicates that the object is placed on the office table. Ignore any spatial relationships not listed in this list.
                Please take note of the following.
                1. Focus only on the objects related to the human action and omit object that are not being manipulated or interacted with in this task. Explain what you included and what you omitted and why in the "your_explanation" field.
                2. The response should be a Python dictionary only, without any explanatory text (e.g., Do not include a sentence like "here is the environment").
                3. Insert "```python" at the beginning and the insert "```" at end of your response.
                """
    PROMPT_MESSAGE = PROMPT_MESSAGE.replace("[ACTION]", instruction)
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [PROMPT_MESSAGE,
                        *map(lambda x: {"image": x}, [frame]),
                        ],
        },
    ]
    if use_azure:
        params = {
            "model": os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
            "messages": PROMPT_MESSAGES,
            "max_tokens": 200,
        }
        client_gpt4v = AzureOpenAI(
            api_version="2024-02-01",
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            api_key=os.environ.get("AZURE_OPENAI_API_KEY")
        )
    else:
        params = {
            "model": "gpt-4-vision-preview",
            "messages": PROMPT_MESSAGES,
            "max_tokens": 200,
        }
        client_gpt4v = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
    result = client_gpt4v.chat.completions.create(**params)
    response_json = extract_json_part(result.choices[0].message.content)
    json_dict = json.loads(response_json, strict=False)
    return json_dict
