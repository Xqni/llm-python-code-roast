from ollama import chat
from .config import (
    OLLAMA_MODEL
)

from .tools import read_files, list_files, get_cwd

# list files in the current folder
# read chat.py in the src folder


class LlamaChat:
    def __init__(self, model=OLLAMA_MODEL) -> None:
        self.model = model
        self.messages = []

        self.available_tools = {
            "read_files": read_files,
            "list_files": list_files,
            "get_cwd": get_cwd,
        }

        self.tool_list = [
            read_files,
            list_files,
            get_cwd
        ]

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

            content = ""
            thinking = ""
            tool_calls = []
            for chunk in response:
                if chunk.message.thinking:
                    thinking += chunk.message.thinking
                    print(
                        f"\033[2m{chunk.message.thinking}\033[0m", end="", flush=True)
                if chunk.message.content:
                    content += chunk.message.content
                    print(chunk.message.content, end="", flush=True)
                if chunk.message.tool_calls:
                    tool_calls.extend(chunk.message.tool_calls)

            self.messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls
            })
            if tool_calls:
                for tool in tool_calls:
                    tool_name = tool.function.name
                    arguments = tool.function.arguments

                    print(
                        f"\n\033[2mSystem: AI is running\ntool: \"{tool_name}\"\narguments: {arguments}\033[0m\n")

                    if function_to_call := self.available_tools.get(tool_name):
                        result = function_to_call(**arguments)

                        self.messages.append({
                            "role": "tool",
                            "content": str(result),
                            "name": tool_name
                        })
            else:
                break
