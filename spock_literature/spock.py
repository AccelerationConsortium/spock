"""Main module."""
from langchain_openai import ChatOpenAI
from spock_literature.utils.Helper_LLM import Helper_LLM
import os
import faiss
from langchain_community.document_loaders import PyPDFLoader
import faiss
import os
import json
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from spock_literature.texts import QUESTIONS, PAPERS_PATH
from langchain_ollama import OllamaLLM
from spock_literature.utils.generate_podcast import generate_audio
from pathlib import Path
from typing import List, Optional, Union
from spock_literature.utils.url_downloader import URLDownloader
from langchain.schema import Document


class Spock(Helper_LLM):  
    """Spock class."""
    
    def __init__(
        self,
        model: str = "llama3.1",
        paper: Optional[Union[Path, str]] = None,
        custom_questions: Optional[List[str]] = None,
        publication_doi: Optional[str] = None,
        publication_title: Optional[str] = None,
        publication_url: Optional[str] = None,
        papers_out = PAPERS_PATH # Make this a path
   
    ):
        """
        Initialize a Spock object.

        Args:
            model (str): Model name. Defaults to "llama3.1".
            paper (Path | str | None): Path to the PDF file locally stored. Defaults to None.
            custom_questions (list[str] | None): List of custom questions. Defaults to None.
            publication_doi (str | None): DOI of the paper to analyze. Defaults to None.
            publication_title (str | None): Title of the paper to analyze. Defaults to None.
        """
        super().__init__(model=model)
        self.paper: Optional[Path] = Path(paper) if paper else None
        self.paper_summary: str = ""
        self.custom_questions: List[str] = custom_questions or []
        self.publication_doi: Optional[str] = publication_doi
        self.publication_title: Optional[str] = publication_title
        self.publication_url: Optional[str] = publication_url
        self.topics: str = ""
        self.questions = QUESTIONS
        

        
    
    
    def download_pdf(self):
        """Download the PDF of a publication."""
    
        from scidownl import scihub_download
        
        if self.publication_doi:

            paper = "https://doi.org/" + self.publication_doi
            paper_type = "doi"
            out = f"{PAPERS_PATH}{self.publication_doi.replace('/','_')}.pdf"
            scihub_download(paper, paper_type=paper_type, out=out)
            
            if not os.path.exists(out):
                raise RuntimeError(f"Failed to download the PDF for the publication with DOI: {self.publication_doi}")
            else:
                self.paper = Path(out)
            
        elif self.publication_title:
            
            paper = self.publication_title
            paper_type = "title"
            out = f"{PAPERS_PATH}{self.publication_title.replace(' ','_')}.pdf"
            scihub_download(paper, paper_type=paper_type, out=out)
            if not os.path.exists(out):
                raise RuntimeError(f"Failed to download the PDF for the publication with title: {self.publication_title}")
            else:
                self.paper = Path(out)
                
        elif self.publication_url:
            # Use Script given
            downloader = URLDownloader(url=self.publication_url, download_path=Path(PAPERS_PATH))
            temp_return = downloader()
            print(temp_return)
            if temp_return != None:
                self.paper = temp_return
            else:
                raise RuntimeError(f"Failed to download the PDF for the publication with URL: {self.publication_url}")
        
    
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
        
        
        
    
    
    
    
    
    
    def summarize(self) -> None:
        from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
        from langchain.chains.llm import LLMChain
        from langchain.chains.combine_documents.stuff import StuffDocumentsChain
        from langchain_openai import ChatOpenAI

        """Return the summary of the publication."""
        if isinstance(self.paper, Document):
            docs = self.paper
        else:
            loader = PyPDFLoader(self.paper)
            docs = loader.load_and_split()

        map_template = """The following is a set of documents
        {docs}
        Based on this list of docs, please identify the main themes 
        Helpful Answer:"""
        map_prompt = PromptTemplate.from_template(map_template)
        
        if isinstance(self.llm, ChatOpenAI):
            llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.2)
            #llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        else:
            llm = OllamaLLM(model="llama3.2:3b", temperature=0.2)
        
        
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
            chunk_size=2500
        )
        split_docs = text_splitter.split_documents(docs) if not isinstance(docs, Document) else text_splitter.split_documents([docs])

        # Invoke the chain and extract the summary
        result = map_reduce_chain.invoke(split_docs)
        self.paper_summary = result['output_text']

            
    def get_topics(self):
        if self.paper_summary == "":
            self.summarize()
        prompt = PromptTemplate(
            template="""Here is it's summary: \n {summary} \n Get the scientific topics that are related to the abstarct above. Ouput only the keywords separated by a '/'. Desired format: Machine Learning/New Materials/NLP""", # Work on the prompt/output of LLM
            input_variables=["summary"]
        )
        chain = prompt | self.llm
        
        #TODO: Continue from here/Add topics
        return chain.invoke({"summary": self.paper_summary}).content if isinstance(self.llm, ChatOpenAI) else chain.invoke({"summary": self.paper_summary})
        
        
        
    
    def __call__(self):
        """Run Spock."""
        self.download_pdf()
        self.add_custom_questions()
        self.scan_pdf() 
        
        # To not rerun the summarize method if the topics are already extracted
        if not self.paper_summary: 
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
            
            
            
    def generate_podcast(self, transcript:bool=False):
        """
        Generate a podcast from the publication.
        """
        audio_file_path, transcript = generate_audio(self.paper)
        if transcript:
            return audio_file_path, transcript  
        return audio_file_path  
    
    
    def format_output(self) -> str:
        """Format the output of the Spock class."""
        output_lines = [
            "📄 Summary of the Publication",
            f"{self.paper_summary}",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "📝 Topics Covered in the Publication"
        ]

        if isinstance(self.topics, list):
            for topic in self.topics:
                output_lines.append(f"• {topic}")
        else:
            output_lines.append(f"{self.topics}")

        output_lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        for question in self.questions:
            output_lines.extend([
                f"❓ Question: {question}",
                f"💡 Answer: {self.questions[question]['output']['response']}",
                f"🔎 Supporting Sentence: {self.questions[question]['output']['sentence']}",
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            ])

        output_text = '\n'.join(output_lines)     
        return output_text

    
    
if __name__ == "__main__":
    spock = Spock(
        model="gpt-4o",
        publication_url="https://www.biorxiv.org/content/10.1101/2024.11.11.622734v1"
    )
    spock()
    print(spock.format_output())
    
    spock = Spock(
        model="gpt-4o",
        publication_url="https://www.nature.com/articles/d41586-024-03714-6"
    )
    spock()
    print(spock.format_output())
    
    spock = Spock(
        model="gpt-4o",
        publication_url="https://www.nature.com/articles/s41467-023-44599-9"
    )
    spock()
    print(spock.format_output())


