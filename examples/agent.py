"""A LangChain agent that answers with recent news and cites its sources.

pip install langchain langchain-anthropic langchain-typesearch
TYPESEARCH_API_KEY=ts_live_... ANTHROPIC_API_KEY=... python examples/agent.py "What changed in EU AI Act enforcement this week?"
"""

import sys

from langchain.agents import create_agent

from langchain_typesearch import TypesearchNewsSearch

agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[TypesearchNewsSearch(max_results=8)],  # reads TYPESEARCH_API_KEY
    system_prompt="Answer with recent news. Cite the source and the link of every fact.",
)

question = " ".join(sys.argv[1:]) or "What changed in EU AI Act enforcement this week?"
result = agent.invoke({"messages": [{"role": "user", "content": question}]})
print(result["messages"][-1].content)
