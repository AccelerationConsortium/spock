import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from time import time
from spock_literature import Spock
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv
import numpy as np
import logging


logging.basicConfig(filename='/home/m/mehrad/brikiyou/scratch/spock/examples/experiment.log', level=logging.INFO)
logging.basicConfig(level=logging.INFO)


# TODO: - nemo benchmark
#       - Nim benchmark
#       - slides 
load_dotenv()

 

TIMES_BENCHMARK = 1

"""
nvidia_api_key = os.getenv("NVIDIA_API_KEY")
if not nvidia_api_key:
    raise Exception("NVIDIA_API_KEY is not set in the environment.")
"""

all_times_ollama = []
spock_ollama = Spock(paper="/home/m/mehrad/brikiyou/scratch/spock_2/spock/examples/data-sample.pdf", model="llama3.3")
for i in range(TIMES_BENCHMARK):
    start_time = time()
    spock_ollama.summarize()
    logging.info(f'iteration {i} of llama3.3 using ollama took {time() - start_time} seconds')
    all_times_ollama.append(time() - start_time)
    
    
# Collect timing data for Chat NVIDIA
"""
spock_nvidia = Spock(paper="data-sample.pdf", model="llama3.3")

spock_nvidia.llm = ChatNVIDIA(
    model="meta/llama-3.3-70b-instruct",
    api_key=nvidia_api_key, 
    temperature=0.2,
)

all_times_nvidia = []
for i in range(TIMES_BENCHMARK):
    start_time = time()
    
    spock_nvidia.summarize()
    logging.info(f'iteration {i} of llama3.3 using nvidia took {time() - start_time} seconds')
    #print(spock_nvidia.format_output())
    all_times_nvidia.append(time() - start_time)

avg_time_ollama = np.mean(all_times_ollama)
avg_time_nvidia = np.mean(all_times_nvidia)


with open('results.txt', 'w') as f:
    f.write(f"Average run times:\n"
    f"Spock Llama: {avg_time_ollama:.4f} sec\n"
    f"Chat NVIDIA: {avg_time_nvidia:.4f} sec")
    f.write(";".join(map(str, all_times_ollama)))
    f.write(";".join(map(str, all_times_nvidia)))


fig = plt.figure(figsize=(12, 10))
gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1])

ax_line = fig.add_subplot(gs[0, :])
ax_line.plot(range(TIMES_BENCHMARK), all_times_ollama, label="Spock Llama", marker='o')
ax_line.plot(range(TIMES_BENCHMARK), all_times_nvidia, label="Chat NVIDIA", marker='x')
ax_line.set_xlabel("Run")
ax_line.set_ylabel("Execution Time (seconds)")
ax_line.set_title("Performance Comparison: Spock Llama vs NIM Model")
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
ax_hist_ollama.set_title("Histogram: Spock (llama3.3 70b 4 GPUs) - summary task")
ax_hist_ollama.grid(True)
ax_hist_ollama.text(0.05, 0.95, f"Average run time: {avg_time_ollama:.4f} sec", transform=ax_hist_ollama.transAxes, fontsize=12,
             verticalalignment='top',
             bbox=dict(facecolor='lightgrey', alpha=0.5, pad=5))

# Bottom right: Histogram for Chat NVIDIA
ax_hist_nvidia = fig.add_subplot(gs[1, 1])
ax_hist_nvidia.hist(all_times_nvidia, bins=10, color='orange', alpha=0.7)
ax_hist_nvidia.set_xlabel("Execution Time (seconds)")
ax_hist_nvidia.set_ylabel("Frequency")
ax_hist_nvidia.set_title("Histogram: NVIDIA llama3.3 70b 4 GPU - summary task")
ax_hist_nvidia.grid(True)
ax_hist_nvidia.text(0.05, 0.95, f"Average run time: {avg_time_nvidia:.4f} sec", transform=ax_hist_nvidia.transAxes, fontsize=12,
             verticalalignment='top',
             bbox=dict(facecolor='lightgrey', alpha=0.5, pad=5))

fig.tight_layout()

current_directory = os.getcwd()  
file_path = os.path.join(current_directory, f"comparison_plot_with_histograms_summary_{TIMES_BENCHMARK}_llama3.3.png")
plt.savefig(file_path)
print(f"Plot saved to: {file_path}")

plt.show()
"""