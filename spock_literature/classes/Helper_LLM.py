from operator import itemgetter
import os
from langchain_community.document_loaders.pdf import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
import faiss
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import faiss
import os
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader, TextLoader
from langchain.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
import json
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
import requests
from scholarly import scholarly


class Helper_LLM:
    def __init__(self,model='llama3',embed_model='mxbai-embed-large', folder_path='db2'):
        import getpass
        import os
        from dotenv import load_dotenv
        load_dotenv()

        if not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")


        self.llm = Ollama(model=model)
        self.oembed = OpenAIEmbeddings(model="text-embedding-3-large")
        self.folder_path = folder_path
        self.vectorstore = None

    
    def chunk_indexing(self, document:str):
        # Check if the document is a valid file path
        #from langchain_experimental.text_splitter import SemanticChunker, RecursiveCharacterTextSplitter

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=95)        
        data = []
        if isinstance(document, str) and os.path.isfile(document):
            try:
                pages = PyPDFLoader(document).load_and_split()
                data = text_splitter.split_documents(pages)                
                print(data)
                #chunk_size = 500
                #chunk_overlap = 20
                sliced_pages = text_splitter.split_documents(data)

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
                chunk_size = 180
                chunk_overlap = 5

            except Exception as e:
                raise RuntimeError(f"Error processing text: {e}")

            
        self.vectorstore = FAISS.from_documents(sliced_pages, self.oembed, )

        #all_splits = text_splitter.split_documents(data)
        #self.vectorstore = Chroma.from_documents(documents=all_splits, embedding=self.oembed, persist_directory=self.folder_path)
        
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
            print(res['result'])
            return res['result']


        else:
            raise Exception("No documents loaded")



