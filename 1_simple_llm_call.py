from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

# Simple one-line prompt
prompt = PromptTemplate.from_template("{question}")

# Load Ollama model (make sure it's pulled already)
model = ChatOllama(model="llama3")

parser = StrOutputParser()

# Chain: prompt → model → parser
chain = prompt | model | parser

# Run it
result = chain.invoke({"question": "What is the capital of India?"})

print(result)