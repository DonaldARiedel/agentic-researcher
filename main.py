from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_openai import ChatOpenAI
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_core.tools import Tool
from langchain.agents import initialize_agent
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# Initialize the language model
llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=api_key,
    model="openai/gpt-3.5-turbo",
    temperature=0.7
)

# Set up a DuckDuckGo web search tool
search = DuckDuckGoSearchRun()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for finding websites and information about top companies"
    )
]

# Create an agent that can reason and use the search tool
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True
)

# What we’re asking the agent to do
query = (
    "Find websites that list the Top 50 companies to work for in 2025. "
    "Summarize what each site includes (such as ranking criteria or notable companies). "
    "Then return 3 useful links that could be scraped for company names."
)

# Run the agent and get the response
response = agent.run(query)

# Show result in terminal
print(response)

# Also save result to markdown file
with open("top_companies_sites.md", "w") as f:
    f.write(f"# Top Companies Research\n\n{response}")