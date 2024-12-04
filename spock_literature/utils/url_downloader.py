import os 
import requests
import re
from pathlib import Path
from bs4 import BeautifulSoup
import logging

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
            # From journal
            self.journals_download()
        else:
            self.__preprint_download()


    def __preprint_download(self):
        response = requests.get(self.url)
        if response.status_code != 200:
            raise ConnectionError(f"Failed to download {self.url}")
        
        content_type = response.headers.get('Content-Type', '')
        if 'application/pdf' not in content_type.lower():
            try: 
                pdf_url = self.find_pdf_link(response.text, self.url)
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

        pdf_name = f"{self.url.split('/')[-1]}.pdf"
        with open(self.download_path/pdf_name, "wb") as file:
            file.write(response.content)
        if os.path.getsize(self.download_path/pdf_name) == 0 or not os.path.exists(self.download_path/pdf_name):
            raise Exception(f"Couldn't download the file: {self.download_path/pdf_name}")
            

    
    @staticmethod
    def find_pdf_link(html_response,url):
        soup = BeautifulSoup(html_response, "html.parser")
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
            
                
    def journals_download(self):
        response = requests.get(self.url)
        pdf_found = False
        if response.status_code != 200:
            raise ConnectionError(f"Failed to access {self.url}")
        
        try: 
            pdf_url = self.find_pdf_link(response.text, self.url)
            pdf_name = pdf_url.split("/")[-1]
            pdf_response = requests.get(pdf_url)
            if pdf_response.status_code == 200:
                with open(self.download_path/pdf_name, 'wb') as f:
                    f.write(pdf_response.content)
                if os.path.getsize(self.download_path) == 0 or not os.path.exists(self.download_path):
                    logger.error(f"Couldn't download the file: {self.download_path}")                    
                logger.info(f"PDF downloaded successfully to {self.download_path}")
                pdf_found = True
                return                
            else:
                logger.error(f"Failed to download PDF from {pdf_url}")
        except ValueError as e:
            logger.error(e)
            logger.info("Trying to extract text from the page")
                    
                  
         # Else, extract the text and try to summarize it or just the abstract 
        
        # If text is not available, raise an error
        
    
        
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



if __name__ == "__main__":
    print("Testing URLDownloader")
    url = "https://www.biorxiv.org/content/10.1101/2024.11.11.622734v1" # no Pdf link 
    download_path = Path("/home/m/mehrad/brikiyou/scratch/spock/spock_literature/utils/")
    downloader = URLDownloader(url, download_path)
    downloader()
    #print(downloader.journals_download())
    print("Downloaded successfully")