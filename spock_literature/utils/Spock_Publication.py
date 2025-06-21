
from typing import Generator, Optional, Dict, Union, Any, List
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

# Download necessary models
download_models(
    output_dir=Path("/home/m/mehrad/brikiyou/scratch/docling_artifacts"),
    with_easyocr=True,
    with_smolvlm=True,
)

os.environ["EASYOCR_MODULE_PATH"] = "/home/m/mehrad/brikiyou/scratch/EasyOCR"

@dataclass(frozen=True, eq=False)
class GScholarPublicationObject:
    title: str
    abstract: str
    author: str
    year: PositiveInt
    url: Optional[HttpUrl] = None
    citation: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *, get_topic: bool = False) -> "GScholarPublicationObject":
        """
        Construct from a raw dict that may contain 'publication_filled'.
        If get_topic=True, you could extend this to call an LLM to assign 'topic'.
        """
        raw = data.get("publication_filled", {})
        bib = raw.get("bib", {})
        year = bib.get("pub_year")
        obj = cls(
            title=bib.get("title", ""),
            abstract=bib.get("abstract", ""),
            author=bib.get("author", ""),
            year=int(year) if year is not None else 0,
            url=raw.get("pub_url"),
            citation=bib.get("citation", ""),
        )
        return obj

    @property
    def summary(self) -> str:
        """A one-line summary combining title and year."""
        return f"{self.title} ({self.year})"

    def to_json(self) -> str:
        """Serialize this object to a JSON string."""
        return json.dumps(self.__dict__)

    @staticmethod
    def from_json(json_str: str) -> "GScholarPublicationObject":
        """
        Deserialize a JSON string to a GScholarPublication object.
        """
        data = json.loads(json_str)
        return GScholarPublicationObject(
            title=data.get("title", ""),
            abstract=data.get("abstract", ""),
            author=data.get("author", ""),
            year=data.get("year", 0),
            url=data.get("url"),
            citation=data.get("citation", ""),
        )

@dataclass(frozen=True, eq=False)
class GScholar_Author(BaseModel):
    def __init__(self, author):
        """
        Initialize an Author object.

        Args:
            author (str): The name of the author.
        """
        self.author_name = author
        
    def __str__(self):
        return self.author_name
    
    def get_last_publications(self, count) -> dict:
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

    def __call__(self, count: int = 1) -> Generator["Publication", None, None]:  
        """
        Get the n last publications of the author.
        """       
        author_publications = self.get_last_publications(count)
        for publication in author_publications:
            yield Publication(publication)
        


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
class SpockPDFLoader(BasePDFLoader):
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
    # TODO: Add support for structured data
    #images: Optional[List[Image]] = Field(default_factory=list)
    #tables: Optional[List[Table]] = Field(default_factory=list)
    
    
    
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
    