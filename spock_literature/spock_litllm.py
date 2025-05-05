import os
import multiprocessing
from time import time
import litellm
import PyPDF2

def _split_text(text, max_chars=5000):
    """Splits text into chunks of at most max_chars characters."""
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

def process_chunk(chunk: str) -> str:
    """
    Summarizes a text chunk using litellm with an improved prompt.
    This prompt instructs the assistant to provide a detailed summary,
    focusing on the main themes, key insights, and critical details.
    """
    prompt = (
        "Please provide a detailed summary of the following text. "
        "Focus on extracting the main themes, key insights, and critical details "
        "that capture the essence of the content.\n\n"
        f"{chunk}"
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    response = litellm.completion(
        model="ollama/llama3.2:3b",
        messages=messages,
        temperature=0.2
    )
    return response["choices"][0]["message"]["content"]

def worker_process(chunks, gpu_id):
    """
    Worker function to process a list of chunks.
    Sets the environment variable for the GPU (CUDA_VISIBLE_DEVICES)
    so that this process uses the assigned GPU.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    results = []
    for chunk in chunks:
        results.append(process_chunk(chunk))
    return results

class Summarizer:
    def __init__(self, paper, model_name: str = "gpt-4", api_key: str = None):
        """
        Parameters:
          paper: Either a PDF file path (str) or raw text.
          model_name: Model used for the merge step.
          api_key: (Optional) API key for litellm.
        """
        self.paper = paper
        self.model_name = model_name
        self.api_key = api_key
        self.paper_summary = None

    def _load_text(self) -> str:
        """Loads text from a PDF file if applicable or returns the text directly."""
        if isinstance(self.paper, str) and self.paper.lower().endswith('.pdf'):
            with open(self.paper, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
            return text
        elif isinstance(self.paper, str):
            return self.paper
        else:
            raise ValueError("Unsupported type for paper. Must be a PDF file path or text string.")

    def summarize(self) -> None:
        start = time()
        full_text = self._load_text()
        chunks = _split_text(full_text, max_chars=5000)

        num_gpus = 4
        groups = [chunks[i::num_gpus] for i in range(num_gpus)]
        
        pool_args = [(group, gpu_id) for gpu_id, group in enumerate(groups)]
        with multiprocessing.Pool(processes=num_gpus) as pool:
            results = pool.starmap(worker_process, pool_args)
        chunk_summaries = [summary for sublist in results for summary in sublist]

        merge_prompt = (
            "You have been provided with several detailed summaries from different portions "
            "of a document. Please integrate these summaries into one coherent and well-structured final summary "
            "that captures the overarching themes, key insights, and critical details of the entire document.\n\n"
            "Summaries:\n" + "\n\n".join(chunk_summaries)
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": merge_prompt}
        ]
        final_response = litellm.completion(
            model=self.model_name,
            messages=messages,
            api_key=self.api_key,
            temperature=0.2
        )
        self.paper_summary = final_response["choices"][0]["message"]["content"]
        print(f"Time taken to summarize the document: {time() - start:.2f} seconds")

if __name__ == "__main__":
    pdf_path = "/home/m/mehrad/brikiyou/scratch/spock_2/spock/examples/data-sample.pdf"
    summarizer = Summarizer(paper=pdf_path, model_name="ollama/llama3.3:70b-instruct-q3_K_M", api_key="your-key")
    summarizer.summarize()
    print(summarizer.paper_summary)
