from ollama import chat
from .config import (
    OLLAMA_MODEL
)

import json

from .tools import read_files, list_files, get_cwd


class LlamaChat:
    def __init__(self, model=OLLAMA_MODEL) -> None:
        self.model = model
        self.messages = []

        self.available_tools = {
            "read_files": read_files,
            "list_files": list_files,
            "get_cwd": get_cwd
        }

        self.tool_list = list(self.available_tools.values())

    def get_response(self, messages):
        self.messages = messages

        print("\033[2m[Qwen thinking...] \033[0m",
              end="", flush=True)

        # Agent Loop
        while True:
            stream = chat(
                model=self.model,
                messages=self.messages,
                tools=self.tool_list,
                stream=True,
            )

            in_thinking = False
            thinking = ""
            content = ""
            tool_calls = None
            for chunk in stream:
                if chunk.message.thinking:
                    if not in_thinking:
                        in_thinking = True
                    # print(
                    #     f"\033[2m{chunk.message.thinking}\033[0m", end="", flush=True)
                    with open("./thinking.txt", "a") as f:
                        f.write(json.dumps(chunk.model_dump(), indent=4))
                    thinking += chunk.message.thinking
                if chunk.message.content:
                    if in_thinking:
                        in_thinking = False
                        print("\nQwen: ", end="", flush=True)
                    # print(f"{chunk.message.content}",
                    #       end="", flush=True)
                    with open("./messageChunks.txt", "a") as f:
                        f.write(json.dumps(chunk.model_dump(), indent=4))
                    content += chunk.message.content
                tool_calls = chunk.message.tool_calls

            if tool_calls:
                self.messages.append(stream.message)

                for tool in tool_calls:
                    tool_name = tool.function.name
                    arguments = tool.function.arguments

                    print(
                        f"\033[2mQwen: [calling {tool_name} with {arguments}]\033[0m", end="\n")

                    function_to_call = self.available_tools.get(tool_name)
                    if function_to_call:
                        result = function_to_call(**arguments)

                        self.messages.append({
                            "role": "tool",
                            "content": str(result),
                            "tool_name": tool_name
                        })
            else:
                self.messages = [{
                    "role": "assistant",
                    "thinking": thinking,
                    "content": content
                }]

            print()
            break
