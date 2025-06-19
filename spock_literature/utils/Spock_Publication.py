
from typing import Optional, Dict, Union, Any, List
from pydantic import (
    BaseModel,
    Field,
    HttpUrl,
    PositiveInt,
)
import json
#import spacy
#from scipy.spatial.distance import cosine
import uuid
import re
import torch
from langchain_core.documents import Document
from pathlib import Path




class ScholarlyObject(BaseModel):
    title: str = Field(..., description="Title of the publication")
    abstract: str = Field(..., description="Abstract of the publication")
    author: str = Field(..., description="Author(s) of the publication")
    year: PositiveInt = Field(..., description="Publication year (positive integer)")
    url: Optional[HttpUrl] = Field(None, description="URL of the publication")
    citation: str = Field(..., description="Formatted citation of the publication")
    topic: Optional[str] = Field(None, description="Topic of the publication, if available")


    @root_validator(pre=True)
    def check_publication_filled(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        # If user passed a raw dict under 'publication_filled', extract fields automatically
        raw = values.get("publication_filled")
        if raw and isinstance(raw, dict):
            bib = raw.get("bib", {})
            values.setdefault("title", bib.get("title", ""))
            values.setdefault("abstract", bib.get("abstract", ""))
            values.setdefault("author", bib.get("author", ""))
            year = bib.get("pub_year")
            if year is not None:
                values.setdefault("year", int(year))
            values.setdefault("citation", bib.get("citation", ""))
            values.setdefault("url", raw.get("pub_url", ""))
        return values
    
    def get_topic(self) -> str:
        """
        Uses an LLM to retrieve the topic of the publication 
        """

    @property
    def summary(self) -> str:
        """A one line summary combining title and year."""
        return f"{self.title} ({self.year})"

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize this object to a JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *, get_topic: bool = False) -> "ScholarlyObject":
        """
        Construct from a raw dict that may contain 'publication_filled'.
        If get_topic=True, you could extend this to call an LLM to assign 'topic'.
        """
        init_kwargs = {"publication_filled": data}
        obj = cls(**init_kwargs)
        if get_topic and obj.topic is None:
            
            pass
        return obj

@dataclass
class Author:
    def __init__(self, author):
        """
        Initialize an Author object.

        Args:
            author (str): The name of the author.
        """
        self.author_name = author
        
    def __repr__(self):
        """
        Return a string representation of the Author object.

        Returns:
            str: The name of the author.
        """
        return self.author_name

    def get_last_publication(self, count) -> dict:
        """
        Get the last publication of the author.

        Returns:
            dict: A dict containing information about the last publication.
        """
        try:
            publications_filled = []
            search_query = scholarly.search_author(self.author_name)
            first_author_result = next(search_query)
            author = scholarly.fill(first_author_result)
            publications = sorted(author['publications'], 
                                    key=lambda x: int(x['bib']['pub_year'])
                                    if 'pub_year' in x['bib'] else 0, 
                                    reverse=True)[0:count]
            for publication in publications:
                publications_filled.append(scholarly.fill(publication))
            return publications_filled
        except Exception as e:
            print(f"An error occurred, couldnt get the latest publications: {e}")

    def __call__(self,count:int=1):
        """
        Setup the author by adding their last publication to a JSON file.

        Args:
            output_file (str): The path to the JSON file.

        Returns:
            data (dict): A dict containing the author's last publication.
        """
        
        data = {}
        author_publications = self.get_last_publication(count)
        for publication in author_publications:
            publication = Publication(publication)
            publication_data = {
                "title": publication.title,
                "abstract": publication.abstract,
                "author": publication.author, 
                "year": publication.year,
                "url": publication.url,
            }
            if self.author_name in data:
                data[self.author_name].append(publication_data)
            else:
                data[self.author_name] = [publication_data]
        return data
    

# TODO: - clean the package
# - 
class Scholarly_Object(BaseModel): # Use pydantic for it   
    publication_filled: Dict[str, Any] = Field(..., description="Filled publication data from the scholarly object")
    title: str = Field(..., description="Title of the publication")
    abstract: str = Field(..., description="Abstract of the publication")
    author : str = Field(..., description="Author of the publication")
    year: str = Field(..., description="Publication year")
    url: str = Field(..., description="URL of the publication")
    citation: str = Field(..., description="Citation of the publication")
    topic: Optional[str] = Field(default=None, description="Topic of the publication, if available")
    
    def __init__(self,publication_filled, get_topic:bool=False) -> None:
        """Initialize a Publication object."""
        self.publication_filled = publication_filled
        self.title = self.get_publication_title()
        self.abstract = self.get_publication_abstract().lower()
        self.author = self.get_author_name()
        self.year = self.get_year()
        self.url = self.get_publication_url()
        self.citation = self.get_citation()
        self.topic = self.get_topic() if get_topic else None
    
      
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
    
    
    
from typing import Optional, Union, List
import json
import time
from pathlib import Path
from docling_core.types.doc import DocItemLabel, ImageRefMode
from docling_core.types.doc.document import DEFAULT_EXPORT_LABELS
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    smolvlm_picture_description,
    AcceleratorDevice,
    AcceleratorOptions,
    EasyOcrOptions,
    PictureDescriptionApiOptions
)
from docling.document_converter import DocumentConverter, PdfFormatOption
import os
import torch
from docling_core.types.doc import PictureItem, ImageRefMode
from langchain_community.document_loaders import BasePDFLoader

os.environ["EASYOCR_MODULE_PATH"] = "/home/m/mehrad/brikiyou/scratch/EasyOCR"



from docling.utils.model_downloader import download_models
download_models(output_dir=Path("/home/m/mehrad/brikiyou/scratch/docling_artifacts"), with_easyocr=True, with_smolvlm=True)

# Plan:
# 1. Use docling for this 
# 2. Use VLM/OCR for parsing
# 3. Output -> Md or else
# 4. Metadata extraction 
# 5. Langchain integration - Document loaders
# 6. Use GPU + parallel processing of everything / Maybe run OCR on CPU 

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
class PDF_document_loader(BasePDFLoader):
    def __init__(self, sources:Union[List[str], List[Path], List[Union[str, Path]]], **kwargs):
        self.sources = sources
        self.num_gpus = torch.cuda.device_count()
        
    def load(self, vlm:Optional[bool]=False, ocr:Optional[bool]=True, **kwargs) -> str:
        """
        Parses Document to markdown or other formats.
        """
        
        scratch = Path("/home/m/mehrad/brikiyou/scratch/docling_artifacts")
        (scratch / "EasyOCR" / "model").mkdir(parents=True, exist_ok=True)
        (scratch / "EasyOCR" / "user_network").mkdir(parents=True, exist_ok=True)

        # Check if 
        if self.num_gpus > 0:
            print(f"Using {self.num_gpus} GPUs for processing.")
            accelerator_options = AcceleratorOptions(
                device=AcceleratorDevice.CUDA,
                num_threads=64,
            )
        else:
            accelerator_options = AcceleratorOptions(
                device=AcceleratorDevice.CPU,
                num_threads=64,
            )
        pipeline_options = PdfPipelineOptions() # artifacts_path=str(scratch)
        pipeline_options.accelerator_options = accelerator_options
        pipeline_options.do_ocr = True
        pipeline_options.do_formula_enrichment = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.do_code_enrichment = True
        
        pipeline_options.ocr_options = EasyOcrOptions(
            use_gpu=(self.num_gpus > 0),
            model_storage_directory=str(scratch / "EasyOCR" / "model"),
           # user_network_directory=str(scratch / "EasyOCR" / "user_network"),
            download_enabled=True
            
        )

        
        if use_vlm:
            if self.num_gpus == 0:
                raise ValueError("No GPUs available for VLM processing.")
            pipeline_options.do_picture_description = True
            pipeline_options.picture_description_options = smolvlm_picture_description # To change to custom function above 
            pipeline_options.picture_description_options.prompt = ("Describe the image in detail and accurately.")                                                                     
            pipeline_options.images_scale = 2.0
            pipeline_options.generate_picture_images = False
            

        converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)
        start  = time.time()
        print(f"Parsing {self.sources[0]}...")
        doc = converter.convert(self.sources[0]).document 
        
        annotations_list = []
        image_counter = 0
        
        for element, *_ in doc.iterate_items():
            if isinstance(element, PictureItem):
                image_counter += 1
                annotation = "\n".join([ann.text for ann in element.annotations]) or "No annotations"
                annotations_list.append(annotation)
        
        output_md_path = '/home/m/mehrad/brikiyou/scratch/spock_2/spock/spock_literature/utils/doc-with-images1.md'
        
        doc.save_as_markdown(
            output_md_path,
            image_mode=ImageRefMode.PLACEHOLDER,
            image_placeholder="%%ANNOTATION%%"
        )
        
        with open(output_md_path, 'r') as file:
            md_content = file.read()
        
        for ann in annotations_list:
            md_content = md_content.replace("%%ANNOTATION%%", ann, 1)
        
        with open(output_md_path, 'w') as file:
            file.write(md_content)


    async def aload(self,):
        raise NotImplementedError("Asynchronous loading is not implemented yet.")
    
    
    @staticmethod
    def to_json(self,) -> dict:
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

       






        
        
class Publication(Document):
    """
    Represents a scientific document with its metadata and content. 
    """
    key:str = Field(..., description="Unique identifier for the document")
    introduction:Optional[str] = Field(default=None, description="Introduction section of the document, introduces the purpose and scope of the research")
    methods:Optional[str] = Field(default=None, description="Methods section of the document")
    results:Optional[str] = Field(default=None, description="Results section of the document")
    discussion:Optional[str] = Field(default=None, description="Discussion section of the document, interprets the results and their implications")
    conclusion:Optional[str] = Field(default=None, description="Conclusion and conclusion section of the document")
    # TODO: Add support for structured data
    # images: Optional[List[Image]] = Field(default_factory=list)
    # tables: Optional[List[Table]] = Field(default_factory=list)
    
    
    
    def extract_sections(self, text:Optional[str]) -> Dict[str, str]:
        """
        Extract sections from the publication content based on markdown-style headings.

        Args:
            text (Optional[str]): The text to extract sections from. If None, uses the page_content of the publication.

        Returns:
            Dict[str, str]: A dictionary where keys are section titles and values are the corresponding section content.
        """
        if not text:
            text = self.page_content
        pattern = re.compile(
            r'(?m)^(##\s*(?:\d+\.\s*)?[^\n]+)\n'    
            r'([\s\S]*?)'                      
            r'(?=^##\s*(?:\d+\.\s*)?[^\n]+|\Z)'
        )

        sections = {}
        for match in pattern.finditer(self.page_content):
            title_line = match.group(1).strip()     
            content    = match.group(2).rstrip()     
            sections[title_line] = content

        return sections

        

        
    def fill_sections(self, text: Union[str, "Publication"]) -> "Publication":
        """
        Fill the sections of the publication with the provided content.
        Uses text clustering to identify and extract sections.
        
        """
        if isinstance(text, Publication):
            source_text = text.page_content
        else:
            source_text = text
            
        if not source_text:
            raise ValueError("Cannot fill sections from empty text")
        
        sections = self.extract_sections(source_text)
        
        # TODO: Implement text clustering logic
        # This would contain your clustering/section identification logic
        
        # For now, return a copy with the source text
        return self.model_copy(update={"page_content": source_text})
    
    @classmethod
    def from_document(cls, doc: Document, **kwargs) -> "Publication":
        return cls(
            key=kwargs.pop("key", getattr(doc, "key", str(uuid.uuid4()))),
            page_content=doc.page_content,
            metadata=doc.metadata,
            **kwargs
        ).fill_sections(doc.page_content)
    
    @classmethod
    def from_pdf(cls, pdf_file:Union[str, Path]):
        """
        """
        
    def get_sections(self) -> Dict[str, Optional[str]]:
        """
        Get all document sections as a dictionary.
        """
        return {
            "introduction": self.introduction,
            "methods": self.methods,
            "results": self.results,
            "discussion": self.discussion,
            "conclusion": self.conclusion,
        }
    
    def has_complete_sections(self) -> bool:
        """
        Check if all major sections are present.
        """
        return all(content is not None and content.strip() for content in self.get_sections.values())
    
    def __repr__(self):
        # To update: Use metadata for title if available
        if self.has_complete_sections():
            return f"Publication(key={self.key}, title={self.metadata.get('title', 'Untitled')})"


if __name__ == "__main__":
    text = Document(
        page_content="""
        ## Introduction
        # """,
        metadata={"title": "Sample Publication"}
    
    )
    
    pub = Publication(text)
    
    

if __name__ == "__main__":
    test_file = [Path("/home/m/mehrad/brikiyou/scratch/spock_2/spock/spock_literature/utils/cell_penetration_of_oxadiazole_containing_macrocycles.pdf")]
    pdf_loader = PDF_document_loader(test_file)
    pdf_loader.parse_document(use_vlm=True)
    