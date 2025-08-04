from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")


prompt = PromptTemplate(
    input_variables=["topic"],
    template="Give me a fun fact about {topic}."
)

# Initialize the language model
llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=api_key,
    model="openai/gpt-3.5-turbo",
    temperature=0.7
)

chain = LLMChain(llm=llm, prompt=prompt)
response = chain.run("basketball")
print(response)