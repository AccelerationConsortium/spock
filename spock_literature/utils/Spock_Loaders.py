
from typing import Generator, Optional, Dict, Union, Any, List, Iterator, AsyncIterator
from pydantic import BaseModel, Field, HttpUrl, PositiveInt
from pydantic.dataclasses import dataclass
import json
import uuid
import time
import os
import torch
from pathlib import Path
from scholarly import scholarly
from langchain_core.documents import Document
from langchain_community.document_loaders import BasePDFLoader
from docling_core.types.doc import PictureItem, ImageRefMode
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    smolvlm_picture_description,
    AcceleratorDevice,
    AcceleratorOptions,
    EasyOcrOptions,
    PictureDescriptionApiOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.utils.model_downloader import download_models
import logging

import requests
import re
from pathlib import Path
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
import getpass
from langchain_core.prompts import PromptTemplate
from langchain.schema import Document
from langchain_openai import ChatOpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Download necessary models
download_models(
    output_dir=Path("/home/m/mehrad/brikiyou/scratch/docling_artifacts"),
    with_easyocr=True,
    with_smolvlm=True,
)

os.environ["EASYOCR_MODULE_PATH"] = "/home/m/mehrad/brikiyou/scratch/EasyOCR"


def configure_vlm_server(use_gpt:bool, vlm_model: str, prompt: str, url: str): 
    """
    Configure options for the OpenAI or Tensort vision-language model API.

    Args:
        model (str): The model name to use for image description generation.
        prompt (str): The prompt to use for image description generation.
        
    Returns:
        PictureDescriptionApiOptions: A configuration object for the VLM API.
    """
    if not use_gpt:
        return PictureDescriptionApiOptions(
            url="http://localhost:11434/v1/chat/completions",  # To change to match Tensorrt server or chatgpt
            params={
                "model": vlm_model,      
                "max_completion_tokens": 512
            },
            prompt=prompt,
            timeout=10,
        )
        
        
def configure_ocr_server(use_easyocr: bool, use_gpu: bool, model_storage_directory: str):
    pass


class SpockPDFLoader(BasePDFLoader):
    def __init__(
        self,
        file_path: Union[str, Path],
        vlm: bool = None,
        ocr: bool = None,
        vlm_prompt: str = "Describe the image in detail and accurately",
        **kwargs
    ):
        """
        

        Args:
            file_path (Union[str, Path]): _description_
            vlm (bool, optional): _description_. Defaults to None.
            ocr (bool, optional): _description_. Defaults to None.
            vlm_prompt (str, optional): _description_. Defaults to "Describe the image in detail and accurately".

        Raises:
            ValueError: _description_
        """
        super().__init__(file_path=file_path, **kwargs)
        self.vlm = vlm
        self.ocr = ocr
        self.num_gpus = torch.cuda.device_count()

        scratch = Path(os.getcwd())
        (scratch / "EasyOCR" / "model").mkdir(parents=True, exist_ok=True)
        (scratch / "EasyOCR" / "user_network").mkdir(parents=True, exist_ok=True)

        if self.num_gpus > 0:
            accelerator_options = AcceleratorOptions(
                device=AcceleratorDevice.CUDA,
                #num_threads=64,
            )
        else:
            accelerator_options = AcceleratorOptions(
                device=AcceleratorDevice.CPU,
                num_threads=64,
            )

        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.accelerator_options = accelerator_options

        self.pipeline_options.do_ocr = self.ocr
        if self.ocr is not None:
            
            if isinstance(self.ocr, EasyOcrOptions):
                self.pipeline_options.ocr_options = self.ocr
            elif isinstance(self.ocr, str) and self.ocr.lower() == "easyocr":
                self.pipeline_options.ocr_options = EasyOcrOptions(
                    use_gpu=(self.num_gpus > 0),
                    model_storage_directory=str(scratch / "EasyOCR" / "model"),
                    download_enabled=True,
                )
            else:
                raise ValueError(
                    "Invalid OCR option. Use 'easyocr' or provide an EasyOcrOptions instance."
                )
                

        self.pipeline_options.do_formula_enrichment = True
        self.pipeline_options.do_table_structure = True
        self.pipeline_options.table_structure_options.do_cell_matching = True
        self.pipeline_options.do_code_enrichment = True

        self.pipeline_options.do_picture_description = self.vlm
        if self.vlm:
            pd_opts = smolvlm_picture_description
            pd_opts.prompt = vlm_prompt 
            self.pipeline_options.picture_description_options = pd_opts
            self.pipeline_options.images_scale = 2.0
            self.pipeline_options.generate_picture_images = False

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options)
            }
        )
        print(f"Parsing {self.sources[0]}...")

    # To check this later
    def lazy_load(self) -> Iterator[Document]:
        """
        Parses Document to markdown or other formats.
        """
        doc = self.converter.convert(self.file_path).document

        annotations_list = []
        for element, *_ in doc.iterate_items():
            if isinstance(element, PictureItem):
                annotation = "\n".join([ann.text for ann in element.annotations]) or "No annotations"
                annotations_list.append(annotation)

        output_md_path = Path("/home/m/mehrad/brikiyou/scratch/spock_2/spock/spock_literature/utils/doc-with-images1.md")
        doc.save_as_markdown(
            output_md_path,
            image_mode=ImageRefMode.PLACEHOLDER,
            image_placeholder="%%ANNOTATION%%"
        )

        md_content = output_md_path.read_text()
        for ann in annotations_list:
            md_content = md_content.replace("%%ANNOTATION%%", ann, 1)
        output_md_path.write_text(md_content)
        
    async def alazy_load(self) -> AsyncIterator[Document]:
        pass

    @staticmethod
    def to_json():
        """
        Gets a md file -> to json + images and stuff (check Ilya Rice's wokr)
        """
        pass

    def create_ocr(self):
        """
        Use EasyOCR to parse the document.
        """
        pass

    def create_vlm(self):
        """
        Use VLM to parse the document.
        """
        pass

       


load_dotenv()
def get_api_key(env_var, prompt):
    
    if not os.getenv(env_var):
        os.environ[env_var] = getpass.getpass(prompt)

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
                pdf_name = pdf_url.split("/")[-1] + ".pdf"  
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
        
        response = requests.get(self.url)
        if response.status_code != 200:
            raise ConnectionError(f"Failed to access {self.url}")
        
        try: 
            data,soup = self.extract_html_text(response.text)           
            document = Document(page_content=data['text'], metadata={"title": data['title'], "headings": data['headings'], "source":""})
            response = self.llm_document_decider(document)
            logger.info(f"Document is a complete scientific paper: {response}")
            if response:
                # If the document is a complete scientific paper
                return document

            pdf_url = self.find_pdf_link(soup, self.url)
            pdf_name = pdf_url.split("/")[-1] + ".pdf"  
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
            
    
        
    @staticmethod
    def validator(url:str) -> bool:
        url_regex = (
            r"^(https?://)" 
            r"(([a-zA-Z0-9\-]+\.)+[a-zA-Z]{2,})" 
            r"(:\d+)?(/.*)?$"
        )
        return bool(re.match(url_regex, url))
    
    
    @staticmethod
    def llm_document_decider(document:Document):


        prompt = PromptTemplate( # Would return a pydantic publication object
            template=f"""
Here is a text, and we need to determine whether it represents a complete scientific article or a sufficiently comprehensive scientific piece (such as a commentary, feature, or news article) that conveys scientific findings or analysis in a coherent and self-contained manner. Traditional full-length research articles often include:

1. **Title**: A clear and descriptive title.
2. **Abstract**: A concise summary of the purpose, methods, main findings, and conclusions.
3. **Introduction**: Background information and context that frame the research question or hypothesis, along with its significance.
4. **Methods (or Materials and Methods)**: A detailed description of how the study was conducted, including experimental design, data collection, and analytical techniques.
5. **Results**: A presentation of the study''s findings, often supported by tables, figures, and statistical analysis.
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

If the text either closely aligns with a full-length research article's structure or is a shorter, self-contained scientific piece that adequately conveys a clear scientific message and context (even if non-traditional in format), output 1.
If it lacks critical information, coherence, or appears clearly incomplete as a scientific piece, output 0.

Your output should contain only one number, no text or additional information.
        """,
            input_variables=["document"]
        )
        get_api_key("OPENAI_API_KEY", "Enter your OpenAI API key: ")
        temp_llm = ChatOpenAI(model="gpt-4o", temperature=0.05)
        chain = prompt | temp_llm
        response = chain.invoke({"document": document})
        if "1" in response.content:
            return True
        return False

    

if __name__ == "__main__":
    test_file = [Path("/home/m/mehrad/brikiyou/scratch/spock_2/spock/spock_literature/utils/cell_penetration_of_oxadiazole_containing_macrocycles.pdf")]
    pdf_loader = PDF_document_loader(test_file)
    pdf_loader.parse_document(use_vlm=True)
    