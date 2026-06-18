from ollama import chat
from .config import (
  OLLAMA_MODEL
)

class LlamaChat:
  def get_response(self, messages):
    stream = chat(
        model=OLLAMA_MODEL,
        messages=messages,
        think=True,
        stream=True,
    )
    
    for chunk in stream:
        print(f"\033[2m[think]\033[0m", end="")
        if chunk.message.thinking:
            print(f"\033[2m[think] {chunk.message.thinking}\033[0m", end="", flush=True)
        if chunk.message.content:
            print("Llama: ", end="")
            print(chunk.message.content, end="", flush=True)
    print()