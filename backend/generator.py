from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate
from typing import AsyncGenerator

from backend.model import model
from backend.system_prompt import system_prompt
from backend.retreiver import retrieve_context

async  def generate_chat(input: str) -> AsyncGenerator[str, None]:
    prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
    agent = create_tool_calling_agent(llm=model, tools=[retrieve_context], prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=[retrieve_context],
        verbose=True,
        handle_parsing_errors=True
    )

    async for event in agent_executor.astream_events({"input": input}, version="v2"):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if hasattr(chunk, "content") and chunk.content:
                yield chunk.content

