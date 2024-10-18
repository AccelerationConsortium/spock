from Publication import Publication

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



class Publication:
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
        self.pdf = self.get_pdf()
        self.topic = None
    
      
      
    '''  
    def get_topic(self,output_file="json/ouput.json", # à voir cette histoire avec get_topic et __get_topic
                  input_file="json/response.json") -> list[str]:
        try:
            with open(output_file,'r') as file:
                data = json.load(file)
            return data[self.author]['topic']
        except Exception as e:
            return self.__get_topic(input_file)
     '''   
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
    
    def get_topic(self,llm):
        self.topic: dict = llm.get_topic_publication()
        return self.topic
    
    def get_pdf(self):
        query = self.title.lower().replace(" ", "+")
        url = f"https://scholar.google.com/scholar?q={query}"
        response = requests.get(url)
        if response.status_code == 200:
            html_content = response.text
            try:
                return self.__parse_google_scholar(html_content)
            except Exception as e:
                print(f"An error occurred while fetching the PDF link: {e}")
        else:
            print(f"Failed to fetch the page. Status code: {response.status_code}")
            return None

    def __repr__(self) -> str:
        self.available_attributes = {'title': self.title if self.title is not None else 'N/A',
                                     'abstract' : self.abstract if self.abstract is not None else 'N/A',
                                        'author': self.author if self.author is not None else 'N/A',
                                        'year': self.year if self.year is not None else 'N/A',
                                        'url': self.url if self.url is not None else 'N/A',
                                        'citation': self.citation if self.citation is not None else 'N/A',
                                        'pdf': self.pdf if self.pdf is not None else 'N/A',
                                        'topic': self.topic if self.topic is not None else 'N/A'}
        return str(self.available_attributes)


        
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
           

        
