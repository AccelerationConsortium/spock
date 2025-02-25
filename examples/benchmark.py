import os
from langchain_ollama import OllamaLLM
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from time import time
from spock_literature import Spock
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv
import numpy as np

# TODO: - nemo benchmark
#       - Nim benchmark
#       - slides 
load_dotenv()



TIMES_BENCHMARK = 30 


nvidia_api_key = os.getenv("NVIDIA_API_KEY")
if not nvidia_api_key:
    raise Exception("NVIDIA_API_KEY is not set in the environment.")

all_times_ollama = []
spock_ollama = Spock(paper="data-sample.pdf", model="llama3.3")
spock_ollama.llm = OllamaLLM(model="llama3.1:latest", temperature=0.2)
spock_ollama.chunk_indexing(spock_ollama.paper)
for i in range(TIMES_BENCHMARK):
    start_time = time()
    spock_ollama.scan_pdf()
    all_times_ollama.append(time() - start_time)
# Collect timing data for Chat NVIDIA

spock_nvidia = Spock(paper="data-sample.pdf", model="llama3.3")
spock_nvidia.llm = ChatNVIDIA(
    model="meta/llama-3.1-8b-instruct",
    api_key=nvidia_api_key, 
    temperature=0.2,
)
spock_nvidia.chunk_indexing(spock_nvidia.paper)
all_times_nvidia = []
for i in range(TIMES_BENCHMARK):
    start_time = time()
    
    spock_nvidia.scan_pdf()
    print(spock_ollama.format_output())
    
    #print(spock_nvidia.format_output())
    all_times_nvidia.append(time() - start_time)

avg_time_ollama = np.mean(all_times_ollama)
avg_time_nvidia = np.mean(all_times_nvidia)


fig = plt.figure(figsize=(12, 10))
gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1])

ax_line = fig.add_subplot(gs[0, :])
ax_line.plot(range(TIMES_BENCHMARK), all_times_ollama, label="Spock Llama", marker='o')
ax_line.plot(range(TIMES_BENCHMARK), all_times_nvidia, label="Chat NVIDIA", marker='x')
ax_line.set_xlabel("Iteration")
ax_line.set_ylabel("Execution Time (seconds)")
ax_line.set_title("Performance Comparison: Spock Llama vs Chat NVIDIA")
ax_line.legend()
ax_line.grid(True)

average_text = (
    f"Average run times:\n"
    f"Spock Llama: {avg_time_ollama:.4f} sec\n"
    f"Chat NVIDIA: {avg_time_nvidia:.4f} sec"
)
ax_line.text(0.05, 0.95, average_text, transform=ax_line.transAxes, fontsize=12,
             verticalalignment='top',
             bbox=dict(facecolor='lightgrey', alpha=0.5, pad=5))

ax_hist_ollama = fig.add_subplot(gs[1, 0])
ax_hist_ollama.hist(all_times_ollama, bins=10, color='blue', alpha=0.7)
ax_hist_ollama.set_xlabel("Execution Time (seconds)")
ax_hist_ollama.set_ylabel("Frequency")
ax_hist_ollama.set_title("Histogram: Spock Llama")
ax_hist_ollama.grid(True)

# Bottom right: Histogram for Chat NVIDIA
ax_hist_nvidia = fig.add_subplot(gs[1, 1])
ax_hist_nvidia.hist(all_times_nvidia, bins=10, color='orange', alpha=0.7)
ax_hist_nvidia.set_xlabel("Execution Time (seconds)")
ax_hist_nvidia.set_ylabel("Frequency")
ax_hist_nvidia.set_title("Histogram: Chat NVIDIA")
ax_hist_nvidia.grid(True)

fig.tight_layout()

current_directory = os.getcwd()  
file_path = os.path.join(current_directory, "comparison_plot_with_histograms.png")
plt.savefig(file_path)
print(f"Plot saved to: {file_path}")

plt.show()
