import os
from crewai import Agent, Task, Crew,LLM

 #LLM Object from crewai package
llm=LLM(model="ollama/gemma3:latest", base_url="http://localhost:11434")


info_agent = Agent( llm=llm, role="info", goal="Give information", backstory="I am a bot that can provide information on a wide range of topics.")

task = Task(description="Provide information on the capital of Austria", agent=info_agent, expected_output="one sentence response")

crew = Crew(
    agents=[info_agent],
    tasks=[task]
)

result = crew.kickoff()

print("############")
print(result)