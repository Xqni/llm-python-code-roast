from ollama import chat
from .config import (
    OLLAMA_MODEL,
    MODEL_DISPLAY_NAME
)

from .helpers.func_names import get_tool_registry


class Pylm:
    def __init__(self, model=OLLAMA_MODEL) -> None:
        self.model = model
        self.messages = []
        self.available_tools = get_tool_registry()
        self.tool_list = list(self.available_tools.values())

    def get_response(self, messages):
        # Initialize messages
        self.messages = messages

        # Initial print
        print(f"\033[2m[{MODEL_DISPLAY_NAME} thinking...] \033[0m", end="\r")

        # Agent Loop
        while True:
            response = chat(
                model=self.model,
                messages=self.messages,
                tools=self.tool_list,
                stream=True,  # makes this a generator
            )

            # Accumulators
            content = ""
            tool_calls = []

            # Flags for printing
            content_started = False
            thinking_started = False

            # streaming the response in the terminal
            for chunk in response:
                if chunk.message.thinking:
                    if not thinking_started:
                        # Prefix for the thinking chunks, printed once
                        print(f"\033[2m{MODEL_DISPLAY_NAME} thinking: \033[0m", end="")
                        thinking_started = True
                    # Dimmed print for thinking
                    print(
                        f"\033[2m{chunk.message.thinking}\033[0m", end="", flush=True)
                if chunk.message.content:
                    if not content_started:
                        # Prefix for actual message chunks, only printed once
                        print(f"\n{MODEL_DISPLAY_NAME}: ", end="")
                        content_started = True
                    content += chunk.message.content
                    print(chunk.message.content, end="", flush=True)
                if chunk.message.tool_calls:
                    # Add tool calls to the tool_calls accumulator
                    tool_calls.extend(chunk.message.tool_calls)

            # Save the tool calls to the history after the 'for' loop ends
            self.messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls
            })

            # Only if the agent called tools
            if tool_calls:
                self.call_tools(tool_calls)

                # Agent loops automatically here if the tool calling has not finished

            # Agent never called any tools or the tool calls have ended
            else:
                print()  # Newline before waiting for user input
                break

    def call_tools(self, tools_to_call):
        for tool in tools_to_call:
            # Extract the details
            tool_name = tool.function.name
            arguments = tool.function.arguments

            print(
                f"\n\033[2m[tool] {tool_name}({arguments})\033[0m\n")

            try:
                # If the function is in the available list of tools, call the function (or tool)
                if function_to_call := self.available_tools.get(tool_name):
                    result = function_to_call(**arguments)

                    # Add the result to the history for final response and tracking
                    self.messages.append({
                        "role": "tool",
                        "content": str(result),
                        "name": tool_name
                    })
                else:
                    tool_doesnt_exist = f"Error calling {tool_name} with {arguments}. Check the tool against available tools and call the correct tool with correct arguments."
                    self.messages.append({
                        "role": "tool",
                        "content": str(tool_doesnt_exist),
                        "name": tool_name
                    })
            except Exception as e:
                error_calling_tool = f"Error calling {tool_name} with {arguments}. Error message: {e}"
                self.messages.append({
                    "role": "tool",
                    "content": str(error_calling_tool),
                    "name": tool_name
                })
