
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains.llm import LLMChain
import logging 
from typing import List, Optional, Union
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


# TODO: to update here, this is a base clase and just temp
# Output parser will split the LLM result into a list of queries
class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  # Remove empty lines


class HypotheticalQuestions(BaseModel):
    """Generate hypothetical questions."""

    questions: List[str] = Field(..., description="List of questions")



class Spock_MultiQueryRetriever(MultiQueryRetriever):
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
    def query_decomposition(self):
        template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
        The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
        Generate multiple search queries related to: {question} \n
        Output (3 queries):"""
        prompt_decomposition = ChatPromptTemplate.from_template(template)
    
    def multi_query(self, question ):
        """
        N questions are generated and combine them into a single query at the end
        """
        template = """You are an AI language model assistant. Your task is to generate five 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines. Original question: {question}"""
        prompt_perspectives = ChatPromptTemplate.from_template(template)


        generate_queries = (
            prompt_perspectives 
            | ChatOpenAI(temperature=0) 
            | StrOutputParser() 
            | (lambda x: x.split("\n"))
        )
        #TODO: continue here, this is a base class and just temp
        
    def rag_fusion(self, question):
        
        template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
        Generate multiple search queries related to: {question} \n
        Output (4 queries):"""
        prompt_rag_fusion = ChatPromptTemplate.from_template(template)
        
    def step_back():
        pass
    
    def hyde():
        pass
    
    
    def query_routing():
        pass

        
        
        
    


        
    
    
