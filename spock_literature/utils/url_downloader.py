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




load_dotenv()
def get_api_key(env_var, prompt):
    
    if not os.getenv(env_var):
        os.environ[env_var] = getpass.getpass(prompt)

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
            return self.__journals_download()
        else:
            return self.__preprint_download()


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
                    return self.download_path/pdf_name               
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
            return self.download_path/pdf_name
    
    
    @staticmethod
    def extract_html_text(html_response):
        soup = BeautifulSoup(html_response, "html.parser")
        title = soup.title.string
        text = soup.get_text()
        headings = []
        for i in range(1, 7): 
            for heading in soup.find_all(f"h{i}"):
                if heading.string: 
                    headings.append(heading.string.strip())

        data = {"title": title, "text": text, "headings": headings}
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
            document = Document(page_content=data['text'], metadata={"title": data['title'], "headings": data['headings']})
            response = self.llm_document_decider(document)
            logger.info(f"Document is a complete scientific paper: {response}")
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
                return self.download_path/pdf_name            # Return path ?
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
        #print("-----------------")
        #print(document)
        #print("-----------------")
        prompt = PromptTemplate(
            template=f"""
Here is a text, and we need to determine whether it represents a complete scientific article or a sufficiently comprehensive scientific piece (such as a commentary, feature, or news article) that conveys scientific findings or analysis in a coherent and self-contained manner. Traditional full-length research articles often include:

1. **Title**: A clear and descriptive title.
2. **Abstract**: A concise summary of the purpose, methods, main findings, and conclusions.
3. **Introduction**: Background information and context that frame the research question or hypothesis, along with its significance.
4. **Methods (or Materials and Methods)**: A detailed description of how the study was conducted, including experimental design, data collection, and analytical techniques.
5. **Results**: A presentation of the study’s findings, often supported by tables, figures, and statistical analysis.
6. **Discussion**: An interpretation of the results, their implications, their relationship to existing literature, and potential limitations.
7. **Conclusions**: A brief recap of the main findings and their broader significance.
8. **References**: A list of all sources cited.

However, not all scientific articles follow the traditional structure. Some scientific pieces—such as brief communications, news features, commentaries, or perspectives—might not have all these sections explicitly labeled. Instead, they may integrate these elements into a narrative that still conveys background, methodology or approach, key findings or points, analysis or interpretation, and references to the broader scientific context.

Your task:
Examine the text inside {{document}} and determine if it provides a coherent, self-contained scientific narrative that includes some combination of the following:
- A defined scientific topic or question
- Background or context to understand the issue
- Some evidence, data, examples, or references that support its main points
- An explanation or interpretation of the implications or significance of the information presented

If the text either closely aligns with a full-length research article’s structure or is a shorter, self-contained scientific piece that adequately conveys a clear scientific message and context (even if non-traditional in format), output 1.
If it lacks critical information, coherence, or appears clearly incomplete as a scientific piece, output 0.

Your output should contain only one number, no text or additional information.
        """,
            input_variables=["document"]
        )
        get_api_key("OPENAI_API_KEY", "Enter your OpenAI API key: ")
        temp_llm = ChatOpenAI(model="gpt-4o", temperature=0.05)
        chain = prompt | temp_llm
        response = chain.invoke({"document": document})
        #print(response.content)
        if "1" in response.content:
            return True
        return False
        
