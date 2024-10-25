
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



class Publication_scholarly(): # Give it Publication as inheritance
    def __init__(self,publication_filled, llm_use:bool=True, is_from_scholarly:bool=True, **kwargs) -> None:
        """
        Args:
            publication_filled (_type_): _description_
            llm_use (bool, optional): _description_. Defaults to True.
            is_from_scholarly (bool, optional): _description_. Defaults to True.
        """
        if is_from_scholarly: self.publication_filled = publication_filled
        self.title = self.get_publication_title()
        self.abstract = self.get_publication_abstract().lower()
        self.author = self.get_author_name()
        self.year = self.get_year()
        self.url = self.get_publication_url()
        self.citation = self.get_citation()
        self.topic = self.get_topic()
    
      
    def get_publication_url(self) -> str:
        return self.publication_filled['pub_url']
    
    def get_publication_title(self) -> str:
        return self.publication_filled['bib']['title'] 

    def get_publication_abstract(self) -> str:
        return self.publication_filled['bib']['abstract']

    def get_author_name(self) -> str:
        return self.publication_filled['bib']['author']

    def get_year(self) -> str:
        return self.publication_filled['bib']['pub_year']
    
    def get_citation(self) -> str:
        return self.publication_filled['bib']['citation']
    
    def get_topic(self) -> str:
        prompt = PromptTemplate(
            template="You are an AI assistant, and here is a document. get the topic of the document. Here it is: \n {document}",
            input_variables=["document"]
        )
        temp_llm = Ollama(model="llama3.1", temperature=0.05)
        chain = prompt | temp_llm
        return chain.invoke({"document": self.abstract})
    


        
    def __parse_google_scholar(self,html_content):

        soup = BeautifulSoup(html_content, 'html.parser')

        a_tags = soup.find_all('a')
        if __name__ == "__main__": print(a_tags)
        try:
            pdf_link = [a['href'] for a in a_tags if 'href' in a.attrs and '.pdf' in a['href']][0]
            print(f"PDF link found: {pdf_link}")
            self.download_pdf(link=pdf_link)
            return pdf_link
        except Exception as e:
            print(f"An error occurred while parsing the PDF link: {e}")
        
        
    def download_pdf(self,path=os.getcwd()+"/pdfs/",link=""):
        
        # Verifier si le path exist sinon le creer
        
        import os
        import requests
                
        
        path = path + self.title + ".pdf"
        if link != "":
            try:
                response = requests.get(link)
                if response.status_code == 200:
                    with open(path, 'wb') as file:
                        file.write(response.content)
                    print(f"PDF successfully downloaded and saved to {path}")
                else:
                    print(f"Failed to download the PDF. HTTP Status Code: {response.status_code}")

            except requests.exceptions.RequestException as e:
                print(f"An error occurred while downloading the PDF: {e}")
        else:
            from scidownl import scihub_download
            #  scidownl by title
            try:
                scihub_download(self.title, paper_type="title", out=path)
            except:
                try:
                    # By URL
                    scihub_download(self.pdf, out=path)
                except:
                    print("Couldn't download the PDF")
           