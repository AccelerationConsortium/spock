
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

    def __call__(self, count: int = 1) -> Generator["Publication", None, None]:  # Iterator here ? 
        """
        Get the n last publications of the author.
        """       
        author_publications = self.get_last_publications(count)
        for publication in author_publications:
            yield Publication(publication)
        

        
# Useless ??
@dataclass(frozen=True, repr=False)
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


