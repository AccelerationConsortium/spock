import json
from langchain_community.llms import Ollama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

class Publication:
    def __init__(self,publication_filled, llm_use:bool=True) -> None:
        self.publication_filled = publication_filled
        
        self.title = self.get_publication_title()
        self.abstract = self.get_publication_abstract().lower()
        self.author = self.get_author_name()
        self.year = self.get_year()
        self.url = self.get_publication_url()
        self.citation = self.get_citation()
        self.topic = self.get_topic() if not llm_use else self.get_topic_LLM()
        
    def get_topic(self,output_file="json/ouput.json", # à voir cette histoire avec get_topic et __get_topic
                  input_file="json/response.json") -> list[str]:
        try:
            with open(output_file,'r') as file:
                data = json.load(file)
            return data[self.author]['topic']
        except Exception as e:
            return self.__get_topic(input_file)
        
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
    
    def get_topic_LLM(self,input_file="json/response.json") -> dict:



        with open(input_file, 'r') as file:
            data = json.load(file)
                    
        llm = Ollama(
        model="llama3")  # assuming you have Ollama installed and have llama3 model pulled with `ollama pull llama3 `
        parser = JsonOutputParser()
        
        new_text = """The output should be formatted as a JSON instance that conforms to the JSON schema below.

        As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
        the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

        Here is the output schema:
        ```
        {"topic": {'Machine Learning: [Keyword1, keyword2, keyword3], 'Batteries: [keyword1, keyword2, keyword3]}
        ```
        """


        prompt = PromptTemplate(
            template="Here is a text: {abstract} Please identify the topics from the following list: {liste}. Note: A single text can belong to multiple topics, so please list all relevant topics.  \n{format_instructions}"
        ,
            input_variables=["abstract","liste","topics"],
            partial_variables={"format_instructions": new_text}
        )


        chain = prompt | llm | parser
        topics = chain.invoke({"abstract": self.abstract, "liste": data.keys()})
        return topics['topic']
        
        
    def __get_topic(self
            , file) -> list[str]:
        
        topics = []
        with open(file, 'r') as file:
            data = json.load(file)
        
        for category, item in data.items():
            for keyword in item['keywords']:
                if keyword in self.abstract:
                    topics.append(category)
                if keyword in self.title:
                    topics.append(category)
    
                    
        return list(set(topics))
                
        
    
        
