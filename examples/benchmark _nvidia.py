#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
# Write results to a txt file
from time import time
from spock_literature import Spock
from langchain_nvidia_ai_endpoints import ChatNVIDIA

client = ChatNVIDIA(
  model="meta/llama-3.3-70b-instruct",
  api_key="$API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC", 
  temperature=0.2,
  top_p=0.7,
  max_tokens=1024,
)

for chunk in client.stream([{"role":"user","content":"Write a limerick about the wonders of GPU computing."}]): 
  print(chunk.content, end="")

  


occurrences = [5, 10, 25, 50, 100]


spock_ollama = Spock(paper="data-sample.pdf")
spock_ollama()
print(spock_ollama.format_output())
