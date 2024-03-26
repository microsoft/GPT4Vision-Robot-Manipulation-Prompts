from openai import OpenAI, AzureOpenAI
import os


def video_understanding(selected_frames, use_azure=True):
    PROMPT_MESSAGE = """These are frames from a video in which a human is doing something. Understand these frames and generate a one-sentence instruction for humans to command these actions to a robot.
                As a reference, the necessary and sufficient human actions are defined as follows:
                HUMAN ACTION LIST
                Grab(arg1): Take hold of arg1.
                Preconditions: Arg1 is in a reachable distance. No object is held (i.e., BEING_GRABBED)
                Postconditions: Arg1 is held (i.e., BEING_GRABBED).

                MoveHand(arg1): Move a robot hand closer to arg1 to allow any actions to arg1. Arg 1 is a description of the hand's destination. For example, "near the table" or "above the box".

                Release(arg1): Release arg1.
                Preconditions: Arg1 is being held (i.e., BEING_GRABBED).
                Postconditions: Arg1 is released (i.e., not BEING_GRABBED).

                PickUp(arg1): Lift arg1.
                Preconditions: Arg1 is being held (i.e., BEING_GRABBED).
                Postconditions: Arg1 is being held (i.e., BEING_GRABBED).

                Put(arg1, arg2): Place arg1 on arg2.
                Preconditions: Arg1 is being held (i.e., BEING_GRABBED).
                Postconditions: Arg1 is being held (i.e., BEING_GRABBED).

                Response should be a sentence in a form of human-to-human communication (i.e., do not directly use the functions). Return only one sentence without including your explanation in the response (e.g., Do not include a sentence like "here are the step-by-step instructions").
                """
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [PROMPT_MESSAGE,
                        *map(lambda x: {"image": x}, selected_frames),
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
    return result.choices[0].message.content
