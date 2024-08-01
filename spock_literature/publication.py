import json
from langchain_community.llms import Ollama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
import requests
from bs4 import BeautifulSoup
try:
    from bot import Bot_LLM
except:
    from .bot import Bot_LLM

import arxiv 
import requests
import os 
import re 
import urllib.request
import json
import re
import requests 
import os 



def standard_url_alphanumeric(input_string):
    allowed_characters = r'[a-zA-Z0-9\-._~:/?#\[\]@!$&\'()*+,;=%]'
    cleaned_string = re.sub(r'[^\w\s-]', ' ', input_string) 
    cleaned_string = re.sub(r'[-\s]+', '-', cleaned_string)
    cleaned_string = cleaned_string.strip("-")
    cleaned_string = "".join(re.findall(allowed_characters, cleaned_string))
    cleaned_string = cleaned_string.lower()
    return cleaned_string

def standard_url_alphanumeric_arxiv(input_string):
    return standard_url_alphanumeric(input_string).strip('-')

def standard_title_alphanumeric_chemarxiv(input_string):
    return standard_url_alphanumeric(input_string).strip('_')

class Publication:
    def __init__(self,publication_filled, llm_use:bool=True) -> None:
        self.publication_filled = publication_filled
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
        try:
            return self.publication_filled['pub_url']
        except:
            return None
    
    def get_publication_title(self) -> str:
        try:
            return self.publication_filled['bib']['title']
        except:
            return None

    def get_publication_abstract(self) -> str:
        try:

            return self.publication_filled['bib']['abstract']
        except:
            return None

    def get_author_name(self) -> str:
        try:

            return self.publication_filled['bib']['author']
        except:
            return None

    def get_year(self) -> str:
        try:
            return self.publication_filled['bib']['pub_year']
        except:
            return None

    def get_citation(self) -> str:
        try:
            return self.publication_filled['bib']['citation']
        except:
            return None

    def get_publication_url(self) -> str:
        try:
            return self.publication_filled['pub_url']
        except:
            return None
        
    def get_topic(self,llm:Bot_LLM,input_file="json/response.json") -> None:
        self.topic: dict = llm.get_topic_publication_abstract(abstract=self.abstract,input_file=input_file)
        return self.topic
    
    def get_pdf(self):
        url = f"https://scholar.google.com/scholar?q={self.title}"
        response = requests.get(url)
        if response.status_code == 200:
            html_content = response.text
            try:
                return self.__parse_google_scholar(html_content)
            except:
                return None


        else:
            print(f"Failed to fetch the page. Status code: {response.status_code}")
            return None


        
    def __parse_google_scholar(self,html_content):

        soup = BeautifulSoup(html_content, 'html.parser')

        a_tags = soup.find_all('a')
        try:
            pdf_link = [a['href'] for a in a_tags if 'href' in a.attrs and '.pdf' in a['href']][0]
            print(f"PDF link found: {pdf_link}")
            return pdf_link
        except:
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

        
    
    def download_pdf(self,path):
        
        import requests
        path = path + self.title + ".pdf"
        
        if self.pdf is not None:
            try:
                response = requests.get(self.pdf)
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
                    try:
                        self.arxiv_downloader(self.title, path)
                    except:
                        pass
                        
                
           
    def arxiv_downloader(self, directory):
        search = arxiv.Search(
            query = self.title, 
            max_results = 1, 
            sort_by = arxiv.SortCriterion.Relevance
        )
        client = arxiv.Client()
        results = client.results(search)
        for result in results:
            if result.title.lower() == self.title.lower():
                print("Article found.")
                url = result.pdf_url 
                self.pdf = url # Maybe to Split the function in 2
                
                
                
                response = requests.get(url)
                os.makedirs(directory, exist_ok = True)
                title = standard_url_alphanumeric_arxiv(self.title)
                with open(f"{title}.pdf", "wb") as f:
                    f.write(response.content)





    def search_chemrxiv_by_title(self, max_results: int = 1):
        search_query = self.title.split()
        chemrxiv_url = f'https://chemrxiv.org/engage/chemrxiv/public-api/v1/items?term={"%20".join(search_query)}&sort=RELEVANT_DESC&limit={max_results}'

        try:
            req = urllib.request.Request(
                url=chemrxiv_url, 
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req) as response:
                s = response.read()
                jsonResponse = json.loads(s.decode('utf-8'))
                dois = [item["item"]["doi"] for item in jsonResponse["itemHits"]]
                urls = ["https://doi.org/" + doi for doi in dois]
                ids = [item["item"]["id"] for item in jsonResponse["itemHits"]]
                titles = [item["item"]["title"].replace("\n", "") for item in jsonResponse["itemHits"]]
                titles = [standard_url_alphanumeric(self.title) for title in titles]
                pdfs = ["https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/" + id + "/original/" for id in ids]
                pdfs = [pdf + title + ".pdf" for pdf, title in zip(pdfs, titles)]
                return titles, pdfs

        except Exception as e:
            return f"An error occurred."

    def pdf_downloader(self, pdf_link, directory):
        os.makedirs(directory, exist_ok = True)
        
        try:
            response = requests.get(pdf_link)
            response.raise_for_status()
            title = standard_title_alphanumeric_chemarxiv(self.title)
            with open(f"{title}.pdf", "wb") as pdf_file:
                pdf_file.write(response.content)

        except Exception as e:
            print(f"Failed to donwload {pdf_link}")
            
        
        
                    
            






