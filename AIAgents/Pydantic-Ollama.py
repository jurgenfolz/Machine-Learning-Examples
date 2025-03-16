from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel  # This assumes pydanticAI exposes an AI helper class
import os


MODEL_ID = 'gemma3:latest'
OLLAMA_SERVER_ENDPOINT = 'http://localhost:11434/v1'

model_ollama = OpenAIModel(model_name=MODEL_ID, base_url=OLLAMA_SERVER_ENDPOINT)

agent = Agent(
    model=model_ollama,
    system_prompt='reply in one sentence')

response = agent.run_sync('Capital of austria?')
print(response)
