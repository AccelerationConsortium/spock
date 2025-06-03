
from typing import Optional, Dict, Union, Any, List
from pydantic import BaseModel, Field
#import spacy
#from scipy.spatial.distance import cosine
import uuid
import re
from sentence_transformers import SentenceTransformer, util
import torch



class Publication(BaseModel):
    """
    Represents a scientific document with its metadata and content. 
    """
    key:str = Field(description="Unique identifier for the document")
    page_content:Optional[str] = Field(default=None, description="Content of the document")
    metadata:Optional[Dict[str, Any]] = Field(default_factory={}, description="Metadata associated with the document")
    introduction:Optional[str] = Field(default=None, description="Introduction section of the document, introduces the purpose and scope of the research")
    methods:Optional[str] = Field(default=None, description="Methods section of the document")
    results:Optional[str] = Field(default=None, description="Results section of the document")
    discussion:Optional[str] = Field(default=None, description="Discussion section of the document, interprets the results and their implications")
    conclusion:Optional[str] = Field(default=None, description="Conclusion and conclusion section of the document")
    #extra:Optional[List[Field]] = Field(default_factory=[], description="Additional fields for future extensions")
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
    def from_text(cls, text: str, key: Optional[str] = str(uuid.uuid4()), **kwargs) -> "Publication":
        """
        Factory method to create a Publication from raw text.
        """
        instance = cls(key=key, page_content=text, **kwargs)
        return instance.fill_sections(text)
    
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
        sections = self.get_sections()
        return all(content is not None and content.strip() for content in sections.values())