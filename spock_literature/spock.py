"""Main module."""
from langchain_experimental.text_splitter import SemanticChunker
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings,OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from spock_literature.texts import QUESTIONS, PAPERS_PATH
from langchain_ollama import OllamaLLM
from spock_literature.utils.Generate_podcast import generate_audio
from pathlib import Path
from typing import List, Optional, Union
from spock_literature.utils.Url_downloader import URLDownloader
from scidownl import scihub_download
import concurrent.futures
from langchain.docstore.document import Document
from time import time
import nvtx
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from spock_literature.utils.pdf_parsing import PDF_document_loader
import getpass
import re
from langchain_community.vectorstores import FAISS
import faiss
import socket

load_dotenv()
def get_api_key(env_var, prompt):
    if not os.getenv(env_var):
        os.environ[env_var] = getpass.getpass(prompt)

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# TODO:
# Make the paper variable work with everthing, not just a path but urls too (maybe have a class or pydantic object representing a document or paper)
class Spock:  
    """Spock class"""
    def __init__(
        self,
        model:str,
        use_tensor_rt:bool = False,
        paper: Optional[Union[Path, str]] = None,
        #publication_doi: Optional[str] = None,
        #publication_title: Optional[str] = None,
        #publication_url: Optional[str] = None,
        custom_questions: Optional[List[str]] = None, # Batch processing of questions
        papers_download_path: Optional[Union[Path, str]] = Path(os.getcwd() + '/papers/'),
        temperature: float = 0.2,
        embed_model = OpenAIEmbeddings(model="text-embedding-3-large"),
        vectorestore_path: Optional[Union[Path, str]] = Path(os.getcwd() + '/vectorstore/'),
        use_semantic_splitting: bool = False,
        max_tokens: int = 3500,
        **kwargs
   
    ):
        """_summary_

        Args:
            model (str): _description_
            use_tensor_rt (bool, optional): _description_. Defaults to False.
            paper (Optional[Union[Path, str]], optional): _description_. Defaults to None.
            custom_questions (Optional[List[str]], optional): _description_. Defaults to None.
            temperature (float, optional): _description_. Defaults to 0.2.
            embed_model (Optional[str], optional): _description_. Defaults to None.
            vectorestore_path (Optional[Union[Path, str]], optional): _description_. Defaults to Path(os.getcwd() + '/vectorstore/').
            use_semantic_splitting (bool, optional): _description_. Defaults to False.
            max_tokens (int, optional): _description_. Defaults to 3500.
        """
        self.embed_model = embed_model
        if use_tensor_rt:
            timeout = 1.0
            port = kwargs.get("trt_port", 8000)
            is_up = False
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(timeout)
                try:
                    sock.connect(("127.0.0.1", port))
                    is_up = True
                except (socket.timeout, ConnectionRefusedError, OSError):
                    return False
                
            if is_up:
                os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1" # Assuming the server is running already
            else:
                # launch trt server
                pass 
            
        if use_semantic_splitting:
            self.text_splitter = SemanticChunker(
                self.embed_model, breakpoint_threshold_type="gradient"
            )
        else:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
            )
        self.llm = OpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        ) 
        self.embed_model = embed_model(model="text-embedding-3-large") if isinstance(embed_model, type) else embed_model # to edit
        self.paper = paper
        if re.match(r"^https?://", str(self.paper)) or isinstance(self.paper, str):
            
            # Download the paper
            pass 
            
        self.vector_store = FAISS(
                            embedding_function=self.embed_model,
                            index=faiss.IndexFlatL2(len(self.embed_model.embed_query("hello world"))),
                            docstore= InMemoryDocstore(),
                            index_to_docstore_id={}
                        )
        self.vector_store.save_local(folder_path=os.getcwd() + "/vectorstore", index_name=self.paper_name) # to edit 

        
        
        
        
        
    @staticmethod
    def download_paper(
        paper: Union[Path, str],
        papers_path: Union[Path, str] = PAPERS_PATH,
        publication_doi: Optional[str] = None,
        publication_title: Optional[str] = None,
        publication_url: Optional[str] = None
    ):
        """_summary_

        Args:
            paper (Union[Path, str]): _description_
            papers_path (Union[Path, str], optional): _description_. Defaults to PAPERS_PATH.
            publication_doi (Optional[str], optional): _description_. Defaults to None.
            publication_title (Optional[str], optional): _description_. Defaults to None.
            publication_url (Optional[str], optional): _description_. Defaults to None.
        """
        pass
        
  
    @nvtx.annotate("Download PDF")
    def download_pdf(self): # Use maybe arxiv here instead of scihub
        """Download the PDF of a publication."""
        if self.publication_doi and not self.paper:

            paper = "https://doi.org/" + self.publication_doi
            paper_type = "doi"
            out = f"{self.papers_path}{self.publication_doi.replace('/','_')}.pdf"
            scihub_download(paper, paper_type=paper_type, out=out)
            
            if not os.path.exists(out):
                raise RuntimeError(f"Failed to download the PDF for the publication with DOI: {self.publication_doi}")
            else:
                self.paper = Path(out)
            
        elif self.publication_title:
            
            paper = self.publication_title
            paper_type = "title"
            out = f"{self.papers_path}{self.publication_title.replace(' ','_')}.pdf"
            scihub_download(paper, paper_type=paper_type, out=out)
            if not os.path.exists(out):
                raise RuntimeError(f"Failed to download the PDF for the publication with title: {self.publication_title}")
            else:
                self.paper = Path(out)
                
        elif self.publication_url:
            downloader = URLDownloader(url=self.publication_url, download_path=Path(self.papers_path))
            temp_return = downloader()
            if temp_return != None:
                self.paper = temp_return
            else:
                raise RuntimeError(f"Failed to download the PDF for the publication with URL: {self.publication_url}")
        
    
    @nvtx.annotate("Question Answering - RAG retrieval + Generation")
    def scan_pdf(self):
        import concurrent.futures
        import threading
        import time
        
        semaphore_size = 2 if isinstance(self.llm, ChatNVIDIA) else 100  # Max semaphore size for non Nvidia models
        api_semaphore = threading.Semaphore(semaphore_size) # Semaphore size 

        """Scan the PDF of a publication."""
        
        if self.vectorstore is None:
            self.chunk_indexing(self.paper)
        
        # Determine which questions to process.
        lim = 0 if self.settings['Questions'] else 10
        keys_to_process = []
        for i, question_key in enumerate(self.questions):
            if i >= lim:
                keys_to_process.append(question_key)

        def __process_question(question_key):
            tries = 0
            max_retries = 3
            wait_time = 2
            temp_response = None
            while tries < max_retries:
                try:
                    # Small delay before starting the API call
                    with api_semaphore:
                        temp_response = self.query_rag(self.questions[question_key]['question'])
                    break  # Success, exit the loop.
                except Exception as e:
                    if "429" in str(e):
                        print(f"429 error encountered for question {question_key}. Retrying in {wait_time} seconds.")
                        time.sleep(wait_time)
                        wait_time *= 2  # Exponential backoff.
                        tries += 1
                    else:
                        print("An error occurred while scanning the PDF for the question:", question_key)
                        print(e)
                        temp_response = "NA/None"
                        break

            if temp_response is None:
                temp_response = "NA/None"

            # If Binary Response is enabled, simply split and return.
            if self.settings['Binary Response']:
                parts = temp_response.split('/')
                if len(parts) < 2:
                    parts.append("None")
                return parts[0], parts[1]
            else:
                tries = 0
                wait_time = 2
                result_text = None
                while tries < max_retries:
                    try:
                        prompt = PromptTemplate(
                            template=(
                                "Here is a text {text}. It contains an answer followed by some extracts from a text "
                                "supporting that answer. The output should look like this: Answer/Supporting answers. "
                                "If the answer is 'no' or there is no Supporting sentence mentioned in the text, output, "
                                "followed by a '/None'"
                            ),
                            input_variables=["text"]
                        )
                        chain = prompt | self.llm
                        with api_semaphore:
                            result = chain.invoke({"text": temp_response})
                        text = result.content if hasattr(result, "content") else result
                        result_text = text
                        break
                    except Exception as e:
                        if "429" in str(e):
                            print(f"429 error during chain.invoke for question {question_key}. Retrying in {wait_time} seconds.")
                            time.sleep(wait_time)
                            wait_time *= 2
                            tries += 1
                        else:
                            print("An error occurred during chain.invoke for question:", question_key)
                            print(e)
                            result_text = "NA/None"
                            break

                if result_text is None:
                    result_text = "NA/None"
                parts = result_text.split('/')
                if len(parts) < 2:
                    parts.append("None")
                return parts[0], parts[1]

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            results = list(executor.map(__process_question, keys_to_process))

        for key, (response, sentence) in zip(keys_to_process, results):
            self.questions[key]['output']['response'] = response
            self.questions[key]['output']['sentence'] = sentence
        
    @nvtx.annotate("Summerize")
    def summarize(self) -> None:
        start = time()

        if isinstance(self.paper, Document):
            docs = [self.paper]
        else:
            loader = PyPDFLoader(self.paper)
            docs = loader.load_and_split()  

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000)
        split_docs = text_splitter.split_documents(docs)

        if isinstance(self.llm, ChatOpenAI):
            #llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.2)
            #os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=2500)
        else:
            llm = OllamaLLM(model="llama3.2:3b", temperature=0.2)

        prompt = PromptTemplate(
            template="Please summarize the following text, focusing on the main themes:\n\n{text}",
            input_variables=["text"]
        )
        chain = prompt | self.llm # to edit

        def process_doc(doc):
            summary = chain.invoke({"text": doc})
            return summary.content if hasattr(summary, "content") else summary

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            chunk_summaries = list(executor.map(process_doc, split_docs))

        prompt = PromptTemplate(
            template="""You are provided with several summaries from different chunks of a document.
    Please merge them into a single, cohesive summary that captures the overall main themes. 

    Summaries:
    {summaries}

    """,
            input_variables=["summaries"]
        )
        os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"
        chain = prompt | self.llm
        summaries_text = "\n".join(chunk_summaries)
        #print("---------------------- \n", summaries_text)
        final_summary = chain.invoke({"summaries": summaries_text})

        print(f"Time taken to summarize the document: {time() - start}")

        self.paper_summary = final_summary.content if hasattr(final_summary, "content") else final_summary
            
    @nvtx.annotate("Topics")
    def get_topics(self):
        if self.paper_summary == "":
            self.summarize()
        prompt = PromptTemplate(
            template="""Here is it's summary: \n {summary} \n Get the scientific topics that are related to the abstarct above. Ouput only the keywords separated by a '/'. Desired format: Machine Learning/New Materials/NLP""", # Work on the prompt/output of LLM
            input_variables=["summary"]
        )
        chain = prompt | self.llm
        result = chain.invoke({"summary": self.paper_summary})
        self.topics = result.content if hasattr(result, "content") else result

    
    def __call__(self):
        """Run Spock."""
        self.download_pdf()
        self.add_custom_metrics()
        self.scan_pdf() 
        
        if not self.paper_summary: 
            
            self.summarize()
            if not self.topics:
                self.get_topics()        
            
    
    def add_custom_metrics(self): # Add custom metrics
        """Add custom questions to the questions dictionary."""
        
        for question in self.custom_questions:
            prompt = PromptTemplate(
                template="""Here is a question, I want you to give me to what topic it is related the most. \ Here is the question you are going to work on: {question}. 
                The output should only contain the topic of the question.\ \
                
                Here are some examples to help you: \
                
                Example input 1: Does the document mention any new or novel materials discovered?\
                Output Example 1: 'new materials'\
                    
                -- \
                Example input 2: Does the document mention any new or novel high-throughput or large-scale screening algorithm, methods or workflow?\                                            
                Output Example 2: 'screening algorithms'\
                    
                
                The output should only contain the topic of the question.
                
                """,
                input_variables=["question"]
            )
            
            chain = prompt | self.llm
            question_topic = chain.invoke({"question":question})
            temp_text = " Answer either 'Yes' or 'No' followed by a '/' then the exact sentence without any changes\
                                                from the document that supports your answer. If the answer is No or If you don't know the answer, say 'NA/None'" if self.settings['Binary Response'] else " Respond to the question followed by a '/' then the exact sentence without any changes\
                                                    from the document that supports your answer. If the answer is No or If you don't know the answer, say 'NA/None'"
            self.questions.update({question_topic.content if not isinstance(question_topic, str) else question_topic :{"question":question+temp_text, "output":{'response':"","sentence":""}}})
            
            
            
    def generate_podcast(self, transcript:bool=False):
        """
        Generate a podcast from the publication.
        """
        audio_file_path, transcript = generate_audio(self.paper)
        if transcript:
            return audio_file_path, transcript  
        return audio_file_path  
    
    
    def format_output(self) -> str:
        """Format the output of the Spock class."""
        output_lines = [
            "📄 Summary of the Publication",
            f"{self.paper_summary}",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "📝 Topics Covered in the Publication"
        ]

        if isinstance(self.topics, list):
            for topic in self.topics:
                output_lines.append(f"• {topic}")
        else:
            output_lines.append(f"{self.topics}")

        output_lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        for question in self.questions:
            output_lines.extend([
                f"❓ Question: {question}",
                f"💡 Answer: {self.questions[question]['output']['response']}",
                f"🔎 Supporting Sentence: {self.questions[question]['output']['sentence']}",
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            ])

        output_text = '\n'.join(output_lines)     
        return output_text
    
    
    @nvtx.annotate("Answer Question")
    def answer_question(self, question:str):
        """
        Answer a question
        """
        if self.vectorstore:
            return self.query_rag(question)
        else:
            self.chunk_indexing(self.paper)
            return self.query_rag(question)
        
        
if __name__ == "__main__":
    load_dotenv()
    from langchain.llms import OpenAI
    start = time()
    spock = Spock(
        model="llama3.3",
        paper="/home/m/mehrad/brikiyou/scratch/spock_2/spock/examples/data-sample.pdf")
    
    spock.llm = ChatOpenAI(
        model="llama3.3_70b_trt_engine",
        max_tokens=3500
    )
    spock()
    print(spock.format_output())
    print("Time taken to run Spock: ", time() - start)
