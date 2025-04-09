import os
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
import faiss
from langchain_community.document_loaders import PyPDFLoader
import os
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
#from langchain_anthropic import ChatAnthropic
from langchain_ollama import OllamaLLM
import getpass
import os
from dotenv import load_dotenv
import nvtx


load_dotenv()
def get_api_key(env_var, prompt):
    
    if not os.getenv(env_var):
        os.environ[env_var] = getpass.getpass(prompt)

class Helper_LLM:
    def __init__(
        self,
        model: str,
        temperature: float = 0.2,
        embed_model=None,   # <--- Use None so we only create it if needed
        folder_path= None
    ):
        # 1. Initialize the LLM
        if model == "gpt-4o":
            get_api_key("OPENAI_API_KEY", "Enter your OpenAI API key: ")
            self.llm = ChatOpenAI(model="gpt-4o", temperature=temperature)
        
        elif model == "llama3.3":
            self.llm = OllamaLLM(model="llama3.3:70b-instruct-q3_K_M", temperature=temperature) # to change to llama3.3
            
        else:
            raise ValueError("Model not supported")

        if embed_model is None:
            get_api_key("OPENAI_API_KEY", "Enter your OpenAI API key for embeddings: ")
            embed_model = OpenAIEmbeddings(model="text-embedding-3-large")

        self.oembed = embed_model
        self.folder_path = folder_path
        self.vectorstore = None

    def chunk_indexing(self, document):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=750, chunk_overlap=95)        
        if not isinstance(document,Document) and os.path.isfile(document):
            pages = PyPDFLoader(document).load_and_split()
            sliced_pages = text_splitter.split_documents(pages)
        else:
            print("Not a PDF file")
            sliced_pages = text_splitter.split_documents([document])

        self.vectorstore = FAISS.from_documents(sliced_pages, self.oembed)
        
    @staticmethod
    @nvtx.annotate(message="Document Embedding")
    def chunk_indexing_html(html,embed_model=None):
        if embed_model is None:
            get_api_key("OPENAI_API_KEY", "Enter your OpenAI API key for embeddings: ")
            embed_model = OpenAIEmbeddings(model="text-embedding-3-large")
        soup = BeautifulSoup(html, 'html.parser')
        title = soup.title.string
        text = soup.get_text()        
        document = Document(page_content=text, metadata={"title": title})
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=750, chunk_overlap=25)
        sliced_documents = text_splitter.split_documents([document])
        return FAISS.from_documents(sliced_documents, embed_model)
        
        
    @nvtx.annotate("RAG - Retrieval & Generation")
    def query_rag(self, question:str):
        if self.vectorstore:
            self.retriever = self.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 10, "fetch_k": 50},
                )

            from langchain.chains import RetrievalQA
            qachain=RetrievalQA.from_chain_type(self.llm,chain_type="stuff", retriever=self.retriever, verbose=False)
            res = qachain.invoke({"query": question})
            return res['result']


        else:
            raise Exception("No documents loaded")



