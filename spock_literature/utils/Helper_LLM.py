from operator import itemgetter
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
#from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
import faiss
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
import os
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import OllamaLLM
import getpass
import os
from dotenv import load_dotenv


load_dotenv()
def get_api_key(env_var, prompt):
    
    if not os.getenv(env_var):
        os.environ[env_var] = getpass.getpass(prompt)

class Helper_LLM:
    def __init__(self,model, temperature:int=0.2, embed_model=OpenAIEmbeddings(model="text-embedding-3-large"), folder_path='db2'):

        if model == "gpt-4o":
            get_api_key("OPENAI_API_KEY", "Enter your OpenAI API key: ")
            self.llm = ChatOpenAI(model="gpt-4o", temperature=temperature)

        elif model == "claude3.5sonnet":
            get_api_key("ANTHROPIC_API_KEY", "Enter your Anthropic API key: ")
            self.llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=temperature)

        elif model == "llama3.3":
            self.llm = OllamaLLM(model="llama3.3", temperature=temperature)
            
        else:
            raise ValueError("Model not supported")
            
            
        self.oembed = embed_model
        self.folder_path = folder_path
        self.vectorstore = None

    def chunk_indexing(self, document):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=750, chunk_overlap=95)        
        data = []
        if not isinstance(document,Document) and os.path.isfile(document):
            pages = PyPDFLoader(document).load_and_split()
            sliced_pages = text_splitter.split_documents(pages)
        else:
            print("Not a PDF file")
            sliced_pages = text_splitter.split_documents([document])

        self.vectorstore = FAISS.from_documents(sliced_pages, self.oembed)
        
    
    @staticmethod
    def chunk_indexing_html(html,embed_model=OpenAIEmbeddings(model="text-embedding-3-large")):
        soup = BeautifulSoup(html, 'html.parser')
        title = soup.title.string
        text = soup.get_text()        
        document = Document(page_content=text, metadata={"title": title})
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=750, chunk_overlap=25)
        sliced_documents = text_splitter.split_documents([document])
        return FAISS.from_documents(sliced_documents, embed_model) # Maybe use self here instead of returning a new instance
        
        
        
    def query_rag(self, question:str) -> None:
        if self.vectorstore:
            #docs = self.vectorstore.similarity_search(question)
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



