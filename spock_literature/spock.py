"""Main module."""
import concurrent.futures
import getpass
import os
import re
import socket
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from time import time
import faiss
import nvtx
from dotenv import load_dotenv
from scidownl import scihub_download
from beartype import beartype


from langchain.docstore.document import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain.storage import InMemoryByteStore
from langchain.retrievers import ParentDocumentRetriever, EnsembleRetriever
from langchain.retrievers.multi_vector import MultiVectorRetriever, SearchType
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models.llms import BaseLLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.docstore.in_memory import InMemoryDocstore

from spock_literature.texts import QUESTIONS, PAPERS_PATH
from spock_literature.utils.Generate_podcast import generate_audio
from spock_literature.utils.Spock_Downloader import URLDownloader
from spock_literature.utils.pdf_parsing import PDF_document_loader
from spock_literature.utils.Spock_MultiQueryRetriever import HypotheticalQuestions
from spock_literature.utils.Spock_Retriever import Spock_Retriever
from spock_literature.utils.Spock_Publication import Publication


# Data has to be in md format which is not so great


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
    @beartype
    def __init__(
        self,
        publication: Publication,
        llm: Union[BaseLLM, BaseChatModel] = ChatOpenAI(model="gpt-4o", temperature=0.2),
        smaller_model: Optional[Union[BaseLLM, BaseChatModel]] = None,
        embed_model: OpenAIEmbeddings = OpenAIEmbeddings(model="text-embedding-3-large"),
        vectorstore: Optional[FAISS] = None,
        retrievers: Optional[List[BaseRetriever]] = None,
        splitter: Optional[Union[TextSplitter, SemanticChunker]] = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP),
        #papers_download_path: Optional[Union[Path, str]] = os.getcwd() + '/papers/',
        vectorstore_path: Optional[Union[Path, str]] = os.getcwd() + '/vectorstore/',
        **kwargs
    ):
        """
        Initialize a Spock instance to load, chunk, embed, and query a scientific publication.

        Args:
            publication (Publication):
                The publication object or metadata representing the paper to be processed.
            llm (Union[BaseLLM, BaseChatModel], optional):
                A high-capacity language model for performing generation and summarization tasks.
                Defaults to a ChatOpenAI instance running “gpt-4o” at temperature 0.2.
            smaller_model (Optional[Union[BaseLLM, BaseChatModel]], optional):
                An optional lightweight model for faster or lower-cost inference on simpler tasks.
                Defaults to None.
            embed_model (OpenAIEmbeddings, optional):
                The embedding model used to convert text chunks into vector representations.
                Defaults to the “text-embedding-3-large” OpenAI Embeddings.
            vectorstore (Optional[FAISS], optional):
                An optional FAISS vector store instance to hold embeddings and support similarity search.
                If None, a new in-memory FAISS index will be created internally.
            retrievers (Optional[List[BaseRetriever]], optional):
                A list of one or more retriever objects (e.g., keyword or vector retrievers) used
                to fetch relevant chunks in response to queries. Defaults to None.
            splitter (Optional[Union[RecursiveCharacterTextSplitter, SemanticChunker]], optional):
                The text-splitting strategy that divides the publication into manageable chunks
                for embedding and retrieval. Defaults to a recursive splitter with
                CHUNK_SIZE and CHUNK_OVERLAP.
            papers_download_path (Optional[Union[Path, str]], optional):
                Filesystem path where downloaded publications (PDFs, etc.) will be saved.
                Defaults to “<cwd>/papers/”.
            vectorstore_path (Optional[Union[Path, str]], optional):
                Directory path for persisting or loading the FAISS vectorstore on disk.
                Defaults to “<cwd>/vectorstore/”.
            **kwargs:
                Additional keyword arguments to pass through to underlying components
                (e.g., custom cache settings, logging options, or other hooks).
        """
        
        self.publication = publication
        self.llm = llm
        self.smaller_model = smaller_model if smaller_model else llm
        self.embed_model = embed_model 
        self.splitter = splitter 
        
        if vectorstore is not None and not isinstance(vectorstore, FAISS):
            raise TypeError("Only FAISS vectorstore is supported. Please provide a valid FAISS instance or None to use the one we provide.")
        
        self.vectorstore = vectorstore if vectorstore else self.create_vectorstore(embed_model=self.embed_model, use_flatip=True, use_hnsw=False, use_ivfflat=False)
        self.vectorstore.save_local(folder_path=os.getcwd() + "/vectorstore", index_name={self.paper_name}) # to check later
        
        self.retriever = Spock_Retriever(
            retrievers=retrievers if retrievers else ParentDocumentRetriever(
                                                    vectorstore=vectorstore,
                                                    docstore=InMemoryDocstore(), # To see if to use 2 different docstores
                                                    child_splitter=self.splitter,
                                                    parent_splitter=RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=150),
                                                )

        )        
        
        # Using TensorRT or any local llm with a compatible Open AI server 
        if "localhost" in self.llm.base_url:
            self.start_tensorrt_server() 


    def start_tensorrt_server(self, port: int = 8000):
        """Start TensorRT server for optimized inference."""
        pass
    
    @staticmethod
    def create_tensort_rt_llm(link_to_trt_server, **kwargs) -> "ChatOpenAI":
        """Create a TensorRT LLM instance."""
        return ChatOpenAI(
            model="llama3.3_70b_trt_engine",
            base_url=link_to_trt_server,
            **kwargs
        )
        
        # TODO: Add verification of the tensort server and if available

    
    @staticmethod
    def create_vectorstore(embed_model, use_flatip:bool=True, use_hnsw:bool=False, use_ivfflat:bool=False) -> FAISS:
        dim = len(embed_model.embed_query("hello world"))  
        if use_flatip:
            return FAISS(
                embedding_function=embed_model,
                index=faiss.IndexFlatIP(dim), 
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
                distance_strategy=DistanceStrategy.COSINE,
            ) 
        elif use_hnsw:
            raise NotImplementedError("Not implemented yet")
            return FAISS(
                embedding_function=embed_model,
                index=faiss.IndexHNSWFlat(1536, 32),  # Example parameters, adjust as needed
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
                normalize_L2=True,
                distance_strategy=DistanceStrategy.COSINE,
            )
        elif use_ivfflat:
            raise NotImplementedError("Not implemented yet")
            quantizer = faiss.IndexFlatL2(1536)
            index = faiss.IndexIVFFlat(quantizer, 1536, 100)  # Example parameters, adjust as needed
            index.train(np.random.rand(1000, 1536).astype(np.float32))
            return FAISS()               
            
            
            
            
    @classmethod
    def _from_loader(
        cls,
        loader_fn,
        loader_arg,
        *,
        llm=None,
        smaller_model=None,
        embed_model=None,
        **kwargs
    ) -> "Spock":
        """
        Internal helper: call loader_fn(loader_arg, **kwargs)
        to get a paper/document, then forward into cls().
        """
        paper = loader_fn(loader_arg, **kwargs)
        # allow overriding llm/embed_model via kwargs, or fall back
        return cls(
            llm=llm,
            smaller_model=smaller_model,
            embed_model=embed_model,
            paper=paper,
            **{k: v for k, v in kwargs.items() if k not in {"llm", "smaller_model", "embed_model"}}
        )

    @classmethod
    def from_url(cls, url: str, **kwargs) -> "Spock":
        """Load a Publication from its URL, then initialize."""
        return cls._from_loader(Publication.from_url, url, **kwargs)

    @classmethod
    def from_doi(cls, doi: str, **kwargs) -> "Spock":
        """Load a Publication from its DOI, then initialize."""
        return cls._from_loader(Publication.from_doi, doi, **kwargs)

    @classmethod
    def from_title(cls, title: str, **kwargs) -> "Spock":
        """Load a Publication by title search, then initialize."""
        return cls._from_loader(Publication.from_title, title, **kwargs)

    @classmethod
    def from_pdf(
        cls,
        pdf_path: Union[str, Path],
        **kwargs
    ) -> "Spock":
        """
        Load a PDF file (or list of Document objects) via your PDF loader,
        pick the first Document if it returns a list, then initialize.
        """
        documents = PDF_document_loader(pdf_path, **kwargs)
        paper = documents[0] if isinstance(documents, list) else documents
        return cls(paper=paper, **kwargs)
    
    
    
    def start_tensort_rt_server(self, path_to_model:Union[str, Path], port: int = 8000):
        """Start the TensorRT server for the LLM."""
        raise NotImplementedError("TensorRT server start is not implemented yet.")
        import subprocess
        try:
            subprocess.run(
                ["trtserver", "--model-repository=/path/to/model/repo", "--port", str(port)],
                check=True
            )
            print(f"TensorRT server started on port {port}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to start TensorRT server: {e}")
            
    def answer_question(
        self,
        question: str,
        reranking: bool = False,
        hybrid_search: bool = False,
        search_algorithm: str = "mmr",
        search_algorithm_kwargs: Optional[dict] = None,
        custom_weights: Optional[Dict[str, float]] = None,
        query_transformation: Optional[bool] = False,
        run_manager=None
    ):
        # Default search kwargs - to change
        default_kwargs = {"k": 10, "fetch_k": 50} if search_algorithm == "mmr" else {"k": 10}
        search_kwargs = search_algorithm_kwargs or default_kwargs
        
        if hybrid_search:
            retriever = self.__create_ensemble_retriever(custom_weights)
            config = {"configurable": {"search_kwargs_faiss": search_kwargs}}
            results = retriever.invoke(question, config=config)
        else:
            # Use the best single retriever or default chunk retriever
            retriever = self.__select_best_retriever(question)
            results = retriever.invoke(question)
        
        if reranking:
            results = self._rerank_results(results, question)
        
        return results

    async def aanswer_question(
        self,
        question: str,
        run_manager=None
    ):
        pass

    def print_retriever_summary(self):
        """Print summary of available retrievers"""
        available = self.get_available_retrievers()
        print("Available Retrievers:")
        print("-" * 50)
        for name, config in available.items():
            print(f"{name.capitalize()}: {config['description']}")
            print(f"  Default weight: {config['weight']}")
        print("-" * 50)
        print(f"Total retrievers: {len(available)}")

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
        
  
        
    def intro_section(self,) -> Dict[str, str]:
        """
        summary + topics and normal stuff +
        """
        summary = self.paper_summary if self.paper_summary else "No summary available."
        topics = self.topics if self.topics else "No topics available."
        hypothetical_questions = ""
        return {"summary": summary,"topics": topics, "questions":hypothetical_questions}
        
    def get_methods_section(self,) -> Dict[str, str]:
        """
        Datasets, Methods, Models used, Screening algorithms, summary of the workflow
        """
        #methods_summary = summarize(self.paper.methods)
        is_method_novel = ""
        datasets = ""
        methods = ""     
        models = ""
        screening_algorithms = ""
        #graph = generate_graph   
        
    def get_conclusion_section(self):
        """
        Matter discovered, discussion, next steps, future work, conclusion
        """
        conclusion_summary = ""
        matter_discovered = ""
        discussion = ""
        
    
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
    
    def answer_general_question():
        """
        Answer a general question asked by the user -> Look at all the vectorestores and retrieve what's best
        """
        pass
    

        
        
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