from operator import itemgetter
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
import faiss
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
import os
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader, TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama



from dotenv import load_dotenv
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

class Helper_LLM:
    def __init__(self,model='nemotron', temperature:int=0.2, embed_model=OpenAIEmbeddings(model="text-embedding-3-large"), folder_path='db2'):
        import getpass
        import os


        self.llm = Ollama(model=model,temperature=temperature)
        self.oembed = embed_model
        self.folder_path = folder_path
        self.vectorstore = None

    
    def chunk_indexing(self, document:str):
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=95)        
        data = []
        if isinstance(document, str) and os.path.isfile(document):
            try:
                pages = PyPDFLoader(document).load_and_split()
                sliced_pages = text_splitter.split_documents(pages)

            except Exception as e:
                raise RuntimeError(f"Error loading PDF: {e}")
        else:
            try:
                # Treat the document as raw text content
                if isinstance(document, str):
                    data.append(Document(page_content=document))
                elif isinstance(document, list):
                    for text in document:
                        data.append(Document(page_content=text))

            except Exception as e:
                raise RuntimeError(f"Error processing text: {e}")

            
        self.vectorstore = FAISS.from_documents(sliced_pages, self.oembed, )

        
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



