"""Main module."""
from spock_literature.utils.Helper_LLM import Helper_LLM
from operator import itemgetter
import os
import faiss
from langchain_community.document_loaders import PyPDFLoader
import faiss
import os
import json
from langchain_core.prompts import PromptTemplate
import requests
from scholarly import scholarly
from spock_literature.utils.docs import LoadDoc
from langchain.text_splitter import RecursiveCharacterTextSplitter
from texts import *
from langchain_ollama import OllamaLLM

class Spock(Helper_LLM): # Heritage a voir plus tard - maybe bot_llm
    """Spock class."""
    def __init__(self, path="",model:str="llama3.1", paper=None, custom_questions:list=[], publication_doi=None, publication_title=None):
        """
        Initialize a Spock object.

        Args:
            paper: Path to pdf file locally stored. Defaults to None.
            custom_questions (list, optional): List of custom questions. Defaults to [].
            publication_doi (_type_, optional): DOI of paper we want to analyze. Defaults to None.
            publication_title (_type_, optional): Title of paper we want to analyze. Defaults to None.
        """        
        super().__init__(model=model)
        self.path = path
        self.paper =  paper # To edit later
        self.paper_summary = ""
        self.custom_questions = custom_questions
        self.publication_doi = publication_doi
        self.publication_title = publication_title
        self.topics = ""
        self.questions = QUESTIONS

        
    
    
    def download_pdf(self):
        """Download the PDF of a publication."""
    
        from scidownl import scihub_download
        
        if self.publication_doi:

            paper = "https://doi.org/" + self.publication_doi
            paper_type = "doi"
            out = f"{self.path}{self.publication_doi.replace('/','_')}.pdf"
            scihub_download(paper, paper_type=paper_type, out=out)
            
            if not os.path.exists(out):
                raise RuntimeError(f"Failed to download the PDF for the publication with DOI: {self.publication_doi}")
            else:
                self.paper = out
            
        elif self.publication_title:
            
            paper = self.publication_title
            paper_type = "title"
            out = f"{self.path}{self.publication_title.replace(' ','_')}.pdf"
            scihub_download(paper, paper_type=paper_type, out=out)
            if not os.path.exists(out):
                raise RuntimeError(f"Failed to download the PDF for the publication with title: {self.publication_title}")
            else:
                self.paper = out
        
    
    def scan_pdf(self):
        """Scan the PDF of a publication."""
        
        self.chunk_indexing(self.paper)
        for question in self.questions:
            try:
                temp_response = self.query_rag(self.questions[question]['question'])
            except Exception as e:
                print("An error occured while scanning the PDF for the question: ", question)
                temp_response = "NA/None"
            print(temp_response)
            temp_response = temp_response.split('/')
            self.questions[question]['output']['response'] = temp_response[0]
            self.questions[question]['output']['sentence'] = temp_response[1]
        
        
        
        
        
    def summarize(self):
        from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
        from langchain.chains.llm import LLMChain
        from langchain.chains.combine_documents.stuff import StuffDocumentsChain
        from langchain_openai import ChatOpenAI

        """Return the summary of the publication."""
        loader = PyPDFLoader(self.paper)
        docs = loader.load_and_split()

        map_template = """The following is a set of documents
        {docs}
        Based on this list of docs, please identify the main themes 
        Helpful Answer:"""
        map_prompt = PromptTemplate.from_template(map_template)
        
        if isinstance(self.llm, OllamaLLM):
            llm = OllamaLLM(model="llama3.1", temperature=0.2)
        
        else: llm = self.llm
        map_chain = LLMChain(llm=llm, prompt=map_prompt)

        reduce_template = """The following is set of summaries:
        {docs}
        Take these and distill it into a final, consolidated summary of the main themes. 
        Helpful Answer:"""
        reduce_prompt = PromptTemplate.from_template(reduce_template)

        reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain, document_variable_name="docs"
        )

        reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=combine_documents_chain,
            collapse_documents_chain=combine_documents_chain,
        )

        map_reduce_chain = MapReduceDocumentsChain(
            llm_chain=map_chain,
            reduce_documents_chain=reduce_documents_chain,
            document_variable_name="docs",
            return_intermediate_steps=False,
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2500, chunk_overlap=20
        )
        split_docs = text_splitter.split_documents(docs)

        # Invoke the chain and extract the summary
        result = map_reduce_chain.invoke(split_docs)
        self.paper_summary = result['output_text']
        print(self.paper_summary)







            
    def get_topics(self):
        if self.paper_summary == "":
            self.summarize()
        prompt = PromptTemplate(
            template="""Here is it's summary: \n {summary} \n Get the scientific topics that are related to the abstarct above. Ouput only the keywords separated by a '/'. Desired format: Machine Learning/New Materials/NLP""", # Work on the prompt/output of LLM
            input_variables=["summary"]
        )
        chain = prompt | self.llm
        
        #TODO: Continue from here/Add topics
        return chain.invoke({"summary": self.paper_summary})
        
        
        
    
    def __call__(self):
        """Run Spock."""
        self.download_pdf()
        self.add_custom_questions()
        self.scan_pdf()
        #print(self.questions)
        self.summarize()
        self.topics = self.get_topics()        
        
        
    
    def add_custom_questions(self):
        """Add custom questions to the questions dictionary."""
        
        print(self.custom_questions)
        for question in self.custom_questions:
            prompt = PromptTemplate(
                template="""Here is a question, I want you to give me to what topic it is related the most. \ Here is the question you are going to work on: {question}. 
                The output should only contain the topic of the question.\ \
                
                Here are some examples to help you: \
                
                Example input 1: Does the document mention any new or novel materials discovered?\
                Output Example 1: 'new materials'\
                    
                -- \
                Example input 2: Does the document mention any new or novel high-throughput or large-scale screening algorithm, methods or workflow?\                                            
                Output Example 2: 'screening algorithms'\
                    
                
                The output should only contain the topic of the question.
                
                """,
                input_variables=["question"]
            )
            
            chain = prompt | self.llm
            question_topic = chain.invoke({"question":question})
            print(f"Custom question: {question_topic}")
            self.questions.update({question_topic:{"question":question+"Answer either 'Yes' or 'No' followed by a '/' then the exact sentence without any changes\
                                                from the document that supports your answer. If the answer is No or If you don't know the answer, say 'NA/None'" , "output":{'response':"","sentence":""}}})
            
            
            
    def generate_podcast(self):
        """
        Generate a podcast from the publication.
        """
        pass
    
    
    def format_output(self) -> str:
        """Format the output of the Spock class."""
        output_lines = [
            "📄 **Summary of the Publication**",
            f"{self.paper_summary}",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "📝 **Topics Covered in the Publication**"
        ]

        if isinstance(self.topics, list):
            for topic in self.topics:
                output_lines.append(f"• {topic}")
        else:
            output_lines.append(f"{self.topics}")

        output_lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        # Iterate over the questions and append formatted strings to the list
        for question in self.questions:
            output_lines.extend([
                f"❓ **Question**: {question}",
                f"💡 **Answer**: {self.questions[question]['output']['response']}",
                f"🔎 **Supporting Sentence**: {self.questions[question]['output']['sentence']}",
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            ])

        # Join all lines into a single string with newline characters
        output_text = '\n'.join(output_lines)
        
        
        #if __name__ == "__main__":
            #print(output_text)
            
        return output_text

    
    


if __name__ == "__main__": # That would be the script to submit for the job
    import sys
    spock = Spock()
    print(len(spock.questions))
