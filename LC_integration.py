from langchain_community.llms import Ollama

# LangChain w/ Phi-2

llm = Ollama(model="phi-2")

# Test Query
response = llm.invoke("Explain the benefits of postpaid mobile plans.")
print(response)