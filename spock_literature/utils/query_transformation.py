
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains.llm import LLMChain
import logging 
from typing import List, Optional, Union

logger = logging.getLogger(__name__)


# TODO: to update here, this is a base clase and just temp
# Output parser will split the LLM result into a list of queries
class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  # Remove empty lines




class MultiQueryTransformation(MultiQueryRetriever):
    """
    A class to handle multiple query transformations.
    """
    def __init__(**kwargs):
        """
        Initializes the MultiQueryTransformation with the given parameters.
        """
        super().__init__(**kwargs)
        
        
        
    # TODO: async method for this too, could be useful 
    def generate_queries(self, question, run_manager):
        pass
        #return super().generate_queries(question, run_manager)


    # TODO: Implement Implemment all the other query mehtods on yt video       
    def __query_decomposition():
        pass

        
    
    
