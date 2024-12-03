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
        with open(self.download_path, "wb") as file:
            file.write(response.content)
        if os.path.getsize(self.download_path) == 0 or not os.path.exists(self.download_path):
            raise Exception(f"Couldn't download the file: {self.download_path}")
                
                
    def journals_download(self):
        response = requests.get(self.url)
        pdf_found = False
        if response.status_code != 200:
            raise ConnectionError(f"Failed to access {self.url}")
        
        soup = BeautifulSoup(response.text, "html.parser")
        for a_tag in soup.find_all("a", href=True):
            href = a_tag['href']
            text = a_tag.get_text(strip=True).lower()
            
            if (
                'pdf' in href.lower() or
                'download pdf' in text or
                href.lower().endswith('.pdf') or
                a_tag.get('title', '').lower().find('pdf') != -1
            ):
                logger.info(f"Found PDF link: {href}")
                pdf_url = requests.compat.urljoin(self.url, href)
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
                    
                  
         # Else, extract the text and try to summarize it or just the abstract 
         
        """
        if not pdf_found:
            logger.error("Couldn't find PDF link")
            logger.info("Trying to extract text from the page")
       
            text = ""
            print(soup.find_all("p"))
            for p_tag in soup.find_all("p"):
                text += p_tag.get_text()
                print(text)
                if text:
                    logger.info(f"Text found: {text}")
                    break
        """
        
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
    url = "https://www.nature.com/articles/s41467-023-44599-9" # no Pdf link 
    download_path = Path("/home/m/mehrad/brikiyou/scratch/spock/spock_literature/utils/")
    downloader = URLDownloader(url, download_path)
    print(downloader.journals_download())
    print("Downloaded successfully")