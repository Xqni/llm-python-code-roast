from ollama import chat
from .config import (
    OLLAMA_MODEL
)

import json

from .tools import read_files, list_files, get_cwd

import requests

# list files in the current folder


class LlamaChat:
    def __init__(self, model=OLLAMA_MODEL) -> None:
        self.model = model
        self.messages = []

        self.available_tools = {
            "read_files": read_files,
            "list_files": list_files,
            "get_cwd": get_cwd,
            "requests": requests.get
        }

        self.tool_list = [read_files, list_files]

    def get_response(self, messages):
        self.messages = messages

        print("\033[2m[Qwen thinking...] \033[0m", end="\r")

        # Agent Loop
        while True:
            response = chat(
                model=self.model,
                messages=self.messages,
                tools=self.tool_list,
                stream=True,
            )

            tool_calls = None
            for chunk in response:
                if chunk.message.thinking:
                    print(chunk.message.thinking, end="", flush=True)
                if chunk.message.content:
                    print(chunk.message.content, end="", flush=True)
                if chunk.message.tool_calls:
                    tool_calls = chunk.message.tool_calls

            if tool_calls:
                for tool in tool_calls:
                    print(f"\n{tool}")
                    tool_name = tool.function.name
                    arguments = tool.function.arguments

                    function_to_call = self.available_tools.get(tool_name)

                    if function_to_call:
                        result = function_to_call(**arguments)

                        self.messages.append({
                            "role": "tool",
                            "content": str(result),
                            "tool_name": tool_name
                        })
            else:
                break
