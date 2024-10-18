"""Main module."""
import json
import time
import concurrent.futures
from publication import Publication
from author import Author
from spock_literature.classes.Helper_LLM import Helper_LLM


class Spock(Helper_LLM): # Heritage a voir plus tard - maybe bot_llm
    """Spock class."""
    def __init__(self):
        """Initialize Spock."""
        super().__init__()
        
    
    
    def download_pdf(self):
        """Download the PDF of a publication."""
        pass
    
    def scan_pdf(self):
        """Scan the PDF of a publication."""
        pass
    
    def 
    
    def chat_with_pdf(self):