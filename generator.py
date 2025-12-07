from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate

from model import model
from system_prompt import system_prompt
from retreiver import retrieve_context


prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
agent = create_tool_calling_agent(llm=model, tools=retrieve_context, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=[retrieve_context],
    verbose=True,
    handle_parsing_errors=True
)


# Use it
response = agent_executor.invoke({"input": "how to calculate taxes?"})
final_answer = response["output"]
print(final_answer)
