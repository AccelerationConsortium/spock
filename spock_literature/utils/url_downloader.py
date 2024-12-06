import os 
import requests
import re
from pathlib import Path
from bs4 import BeautifulSoup
import logging
import os
from dotenv import load_dotenv
import getpass
from langchain_core.prompts import PromptTemplate
from langchain.schema import Document
from langchain_openai import ChatOpenAI




logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class URLDownloader:
    def __init__(self, url: str, download_path: Path):
        if not self.validator(url):
            raise ValueError(f"Invalid URL: {url}")
        self.url = url
        self.download_path = download_path


    def __call__(self) -> None:
        preprint_regex = (
            r"^https://(www\.)?" 
            r"(arxiv\.org|chemrxiv\.org|biorxiv\.org|medrxiv\.org)"  
            r"(/.*)?$"  
        )
        
        if not re.match(preprint_regex, self.url):
            self.__journals_download()
        else:
            self.__preprint_download()


    def __preprint_download(self):
        response = requests.get(self.url)
        if response.status_code != 200:
            raise ConnectionError(f"Failed to download {self.url}")
        
        content_type = response.headers.get('Content-Type', '')
        if 'application/pdf' not in content_type.lower():
            try: 
                data,soup = self.extract_html_text(response.text)                
                pdf_url = self.find_pdf_link(soup, self.url)
                pdf_name = pdf_url.split("/")[-1]
                pdf_response = requests.get(pdf_url)
                if pdf_response.status_code == 200:
                    with open(self.download_path/pdf_name, 'wb') as f:
                        f.write(pdf_response.content)
                    if os.path.getsize(self.download_path) == 0 or not os.path.exists(self.download_path):
                        logger.error(f"Couldn't download the file: {self.download_path}")                    
                    logger.info(f"PDF downloaded successfully to {self.download_path}")
                    return                
                else:
                    logger.error(f"Failed to download PDF from {pdf_url}")
            except ValueError as e:
                logger.error(e)
                logger.info("Trying to extract text from the page")
                
        else: 
            # Content is a PDF
            pdf_name = self.url.split("/")[-1]
            with open(self.download_path/pdf_name, 'wb') as f:
                f.write(response.content)
            if os.path.getsize(self.download_path) == 0 or not os.path.exists(self.download_path):
                logger.error(f"Couldn't download the file: {self.download_path}")                    
            logger.info(f"PDF downloaded successfully to {self.download_path}")
            return
    
    
    @staticmethod
    def extract_html_text(html_response):
        soup = BeautifulSoup(html_response, "html.parser")
        title = soup.title.string
        text = soup.get_text()        
        data = {"title": title, "text": text}
        return data, soup        

    
    @staticmethod
    def find_pdf_link(soup,url):
        #soup = BeautifulSoup(html_response, "html.parser")
        for a_tag in soup.find_all("a", href=True):
            href = a_tag['href']
            text = a_tag.get_text(strip=True).lower()
            
            if (
                'pdf' in href.lower() or
                'download pdf' in text or
                href.lower().endswith('.pdf') or
                a_tag.get('title', '').lower().find('pdf') != -1
            ):
                logger.info(f"Found PDF link: {href.strip('+html')}")
                logger.info(requests.compat.urljoin(url, href.strip('+html')))
                return requests.compat.urljoin(url, href.strip('+html'))
        raise ValueError("Couldn't find PDF link")
            
                
    def __journals_download(self):
        
        # To update 
        response = requests.get(self.url)
        if response.status_code != 200:
            raise ConnectionError(f"Failed to access {self.url}")
        
        try: 
            data,soup = self.extract_html_text(response.text)           
            document = Document(page_content=data['text'], metadata={"title": data['title']})
            response = self.llm_document_decider(document)
            if response:
                # If the document is a complete scientific paper
                return document

            # If the document is not a complete scientific paper / Download the PDF
            pdf_url = self.find_pdf_link(soup, self.url)
            pdf_name = pdf_url.split("/")[-1]
            pdf_response = requests.get(pdf_url)
            if pdf_response.status_code == 200:
                with open(self.download_path/pdf_name, 'wb') as f:
                    f.write(pdf_response.content)
                if os.path.getsize(self.download_path) == 0 or not os.path.exists(self.download_path):
                    logger.error(f"Couldn't download the file: {self.download_path}")                    
                logger.info(f"PDF downloaded successfully to {self.download_path}")
                return                
            else:
                logger.error(f"Failed to download PDF from {pdf_url}")
        except ValueError as e:
            logger.error(e)
            logger.info("Trying to extract text from the page")
            
    
        
    @staticmethod
    def validator(url:str) -> bool:
        url_regex = (
            r"^(https?://)" 
            r"(([a-zA-Z0-9\-]+\.)+[a-zA-Z]{2,})" 
            r"(:\d+)?(/.*)?$"
        )
        if re.match(url_regex, url):
            return True
        return False
    
    
    @staticmethod
    def llm_document_decider(document:Document):
        prompt = PromptTemplate(
            template=f"""
Here is an text, and we need to determine whether it represents a complete scientific paper. To do this, carefully review the text within the `{{document}}` placeholder. A complete scientific article often includes several key components, although formatting and exact naming of sections can vary. For instance, a typical full-length research article might include:

1. **Title and Authors**: A clear and descriptive title, along with the names and affiliations of the authors.
2. **Abstract**: A concise summary of the purpose, methods, main findings, and conclusions of the study.
3. **Introduction**: Background information and context that frame the research question or hypothesis, along with its significance.
4. **Methods (or Materials and Methods)**: A detailed description of how the study was conducted, including experimental design, data collection, and analytical techniques.
5. **Results**: A presentation of the study’s findings, often supported by tables, figures, and statistical analysis.
6. **Discussion**: An interpretation of the results, their implications, their relationship to existing literature, and potential limitations.
7. **Conclusions**: A brief recap of the main findings and their broader significance.
8. **References**: A list of all sources cited in the paper.

Some articles may also include acknowledgments, funding information, appendices, or supplementary materials. The presence or absence of these sections—and the level of completeness in each—helps determine whether the provided text can be considered a full scientific article.

Your task: Examine the text inside {document}. Identify if it contains recognizable sections that align with the structure of a complete scientific article (title, abstract, introduction, methods, results, discussion, conclusion, references) output 1. If it lacks critical sections or clearly appears to be incomplete, output 0
The output should only contain one number, no text or additional information."""
,
            input_variables=["document"]
        )
        
        temp_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.05)
        chain = prompt | temp_llm
        response = chain.invoke({"document": document})
        if "1" in response:
            return True
        return False
        


if __name__ == "__main__":
    print("Testing URLDownloader")
    url = "https://www.biorxiv.org/content/10.1101/2024.11.11.622734v1" # no Pdf link 
    download_path = Path("/home/m/mehrad/brikiyou/scratch/spock/spock_literature/utils/")
    downloader = URLDownloader(url, download_path)
    downloader()
    downloader = URLDownloader("https://www.nature.com/articles/d41586-024-03714-6#author-0", download_path)
    downloader()
    downloader = URLDownloader("https://www.nature.com/articles/s41467-023-44599-9", download_path)
    downloader()
    
    #print(downloader.journals_download())
    print("Downloaded successfully")