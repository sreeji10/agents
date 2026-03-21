import os

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

load_dotenv()

model = OpenAIChatModel(
    "z-ai/glm5",
    provider=OpenAIProvider(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("api_key"),
    ),
)
agent = Agent(model)
...
# Define a very simple agent including the model to use, you can also set the model when running the agent.
agent = Agent(
    model=model,  # Register static instructions using a keyword argument to the agent.
    # For more complex dynamically-generated instructions, see the example below.
    instructions="Be concise, reply with one sentence.",
)

# result = agent.run_sync('Where does "hello world" come from?')
# print(result.output)

app = agent.to_web()
