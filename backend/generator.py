from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate
from typing import AsyncGenerator

from backend.utils import get_llm
from backend.system_prompt import system_prompt
from backend.retreiver import retrieve_context

async def generate_chat(input: str) -> AsyncGenerator[str, None]:
    prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
    
    llm = get_llm()
        
    agent = create_tool_calling_agent(llm=llm, tools=[retrieve_context], prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=[retrieve_context],
        verbose=True,
        handle_parsing_errors=True
    )

    # Track if async is working
    has_content = False

    async for event in agent_executor.astream_events({"input": input}, version="v2"):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if hasattr(chunk, "content") and chunk.content:
                has_content = True
                yield chunk.content

    # Fallback: if nothing was yielded at all
    if not has_content:
        try:
            result = await agent_executor.ainvoke({"input": input})
            output = result.get("output", "")
            if output:
                yield output
        except Exception as e:
            yield f"Error generating response: {str(e)}"

