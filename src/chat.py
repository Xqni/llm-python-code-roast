from ollama import chat
from .config import (
    OLLAMA_MODEL
)

from .tools import read_files, list_files


class LlamaChat:
    def __init__(self, model=OLLAMA_MODEL) -> None:
        self.model = model
        self.messages = []

        self.available_tools = {
            "read_files": read_files,
            "list_files": list_files
        }

        self.tool_list = [read_files, list_files]

    def get_response(self, messages):
        self.messages.extend(messages)

        print(f"\033[2m[think] \033[0m", end="\r")

        # Agent Loop
        while True:
            response = chat(
                model=self.model,
                messages=self.messages,
                tools=self.tool_list,
                think=True,
            )

            # model used a tool
            if response.message.tool_calls:
                # Add the model's tool request to history so it remembers
                self.messages.append(response.message)
                
                for tool in response.message.tool_calls:
                    tool_name = tool.function.name
                    arguments = tool.function.arguments

                    print(f"\033[2m[Qwen] {tool_name} with {arguments}\033[0m", end="\r")

                    function_to_call = self.available_tools.get(tool_name)
                    if function_to_call:
                        result = function_to_call(**arguments)

                        self.messages.append({
                            "role": "tool",
                            "content": str(result),
                            "tool_name": tool_name
                        })

                # The loop auto restarts here!
                # It sends the new history (with the tool results) back to the model.

            # the model did not call a tool
            else:
                # model has the info it needs and is talking to the user
                print("\n" + response.message.content + "\n")
                # save the final answer to the history
                self.messages.append(response.message)
                # breaking out of the infinite while loop
                break



            # stream it later
            # stream = chat(
            #     model=OLLAMA_MODEL,
            #     messages=messages,
            #     think=True,
            #     stream=True
            # )

            # thinking_phase = False
            # response_started = False

            # for chunk in stream:
            #     if chunk.message.thinking:
            #         if not thinking_phase:
            #             print(f"\033[2m[think] \033[0m", end="")
            #             thinking_phase = True
            #         print(f"\033[2m{chunk.message.thinking}\033[0m",
            #               end="", flush=True)
            #     if chunk.message.content:
            #         if not response_started:
            #             if thinking_phase:
            #                 print()
            #             print("Llama: ", end="")
            #             response_started = True
            #         print(chunk.message.content, end="", flush=True)
            # print()
