"""Main module."""
from langchain_openai import ChatOpenAI
from spock_literature.utils.Helper_LLM import Helper_LLM
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from spock_literature.texts import QUESTIONS, PAPERS_PATH
from langchain_ollama import OllamaLLM
from spock_literature.utils.Generate_podcast import generate_audio
from pathlib import Path
from typing import List, Optional, Union
from spock_literature.utils.Url_downloader import URLDownloader
from langchain.schema import Document
from scidownl import scihub_download
import concurrent.futures
from langchain.docstore.document import Document
from time import time



def verificator(paper, publication_doi, publication_title, publication_url):
    """
    Verify if input is valid
    """
    if not paper and not publication_doi and not publication_title and not publication_url:
        raise ValueError("Please provide a paper, publication DOI, publication title, or publication URL.")
    if paper and (publication_doi or publication_title or publication_url):
        raise ValueError("Please provide either a paper or a publication DOI, title, or URL, not both.")
    if publication_doi and (publication_title or publication_url):
        raise ValueError("Please provide either a publication DOI or a publication title or URL, not both.")
    if publication_title and (publication_doi or publication_url):
        raise ValueError("Please provide either a publication title or a publication DOI or URL, not both.")
    if publication_url and (publication_doi or publication_title):
        raise ValueError("Please provide either a publication URL or a publication DOI or title, not both.")
    if os.path.exists(paper) and not paper.endswith(".pdf"):
        raise ValueError("Please provide a PDF file.")

class Spock(Helper_LLM):  
    """Spock class."""
    
    def __init__(
        self,
        model: str = "llama3.3",
        paper: Optional[Union[Path, str]] = None,
        custom_questions: Optional[List[str]] = None,
        publication_doi: Optional[str] = None,
        publication_title: Optional[str] = None,
        publication_url: Optional[str] = None,
        papers_download_path: str = PAPERS_PATH,
        temperature: float = 0.2,
        embed_model=None,
        folder_path=None,
        settings:Optional[dict[str, bool]]={'Summary':True, 'Questions':True,'Binary Response':True}
   
    ):
        """
        Initialize a Spock object.

        Args:
            model (str): Model name. Defaults to "llama3.1".
            paper (Path | str | None): Path to the PDF file locally stored. Defaults to None.
            custom_questions (list[str] | None): List of custom questions. Defaults to None.
            publication_doi (str | None): DOI of the paper to analyze. Defaults to None.
            publication_title (str | None): Title of the paper to analyze. Defaults to None.
            publication_url (str | None): URL of the paper to analyze. Defaults to None.
            papers_download_path (str): Path to download the papers. 
            temperature (float): Temperature for the model. Defaults to 0.2.
            embed_model (str): Embedding model. Defaults to None.
            folder_path (str): Folder path. Defaults to None.
        """
        verificator(paper, publication_doi, publication_title, publication_url)
        super().__init__(model=model, temperature=temperature, embed_model=embed_model, folder_path=folder_path)
        self.paper: Optional[Path] = Path(paper) if paper else None
        self.paper_summary: str = ""
        self.custom_questions: List[str] = custom_questions or []
        self.publication_doi: Optional[str] = publication_doi
        self.publication_title: Optional[str] = publication_title
        self.publication_url: Optional[str] = publication_url
        self.topics: str = ""
        self.settings = settings
        self.questions = QUESTIONS
        self.papers_path = papers_download_path
  
    
    def download_pdf(self):
        """Download the PDF of a publication."""
    
        
        if self.publication_doi and not self.paper:

            paper = "https://doi.org/" + self.publication_doi
            paper_type = "doi"
            out = f"{self.papers_path}{self.publication_doi.replace('/','_')}.pdf"
            scihub_download(paper, paper_type=paper_type, out=out)
            
            if not os.path.exists(out):
                raise RuntimeError(f"Failed to download the PDF for the publication with DOI: {self.publication_doi}")
            else:
                self.paper = Path(out)
            
        elif self.publication_title:
            
            paper = self.publication_title
            paper_type = "title"
            out = f"{self.papers_path}{self.publication_title.replace(' ','_')}.pdf"
            scihub_download(paper, paper_type=paper_type, out=out)
            if not os.path.exists(out):
                raise RuntimeError(f"Failed to download the PDF for the publication with title: {self.publication_title}")
            else:
                self.paper = Path(out)
                
        elif self.publication_url:
            downloader = URLDownloader(url=self.publication_url, download_path=Path(self.papers_path))
            temp_return = downloader()
            if temp_return != None:
                self.paper = temp_return
            else:
                raise RuntimeError(f"Failed to download the PDF for the publication with URL: {self.publication_url}")
        
    
    def scan_pdf(self):
        """Scan the PDF of a publication."""
        
        ########### - to check first if the document is already chunked (self.vectorstore)
        if self.vectorstore == None:
            self.chunk_indexing(self.paper)
        ###########
        
        
        lim = 0 if self.settings['Questions'] else 10

        keys_to_process = []
        for i, question_key in enumerate(self.questions):
            if i >= lim:
                keys_to_process.append(question_key)

        def __process_question(question_key):
            try:
                temp_response = self.query_rag(self.questions[question_key]['question'])
            except Exception as e:
                print("An error occurred while scanning the PDF for the question:", question_key)
                temp_response = "NA/None"


            ###### To delete binary response
            if self.settings['Binary Response']:
                parts = temp_response.split('/')
                if len(parts) < 2:
                    parts.append("None")
                return parts[0], parts[1]
            else:
                try:                    
                    prompt = PromptTemplate(
                        template=(
                            "Here is a text {text}. It contains an answer followed by some extracts from a text "
                            "supporting that answer. The output should look like this: Answer/Supporting answers. "
                            "If the answer is 'no' or there is no Supporting sentence mentioned in the text, output, "
                            "followed by a '/None'"
                        ),
                        input_variables=["text"]
                    )
                    chain = prompt | self.llm
                    
                    result = chain.invoke({"text": temp_response})
                    text = result.content if hasattr(result, "content") else result
                    parts = text.split('/')
                    if len(parts) < 2:
                        parts.append("None")
                    return parts[0], parts[1]
                except Exception as e:
                    print("An error occurred while scanning the PDF for the question:", question_key)
                    return "NA", "None"
                
            #########

        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(__process_question, keys_to_process))

        for key, (response, sentence) in zip(keys_to_process, results):
            self.questions[key]['output']['response'] = response
            self.questions[key]['output']['sentence'] = sentence
        
    
    def summarize(self) -> None:
        start = time()

        if isinstance(self.paper, Document):
            docs = [self.paper]
        else:
            loader = PyPDFLoader(self.paper)
            docs = loader.load_and_split()  

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000)
        split_docs = text_splitter.split_documents(docs)

        if isinstance(self.llm, ChatOpenAI):
            #llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.2)
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        else:
            llm = OllamaLLM(model="llama3.2:3b", temperature=0.2)

        prompt = PromptTemplate(
            template="Please summarize the following text, focusing on the main themes:\n\n{text}",
            input_variables=["text"]
        )
        chain = prompt | llm

        def process_doc(doc):
            summary = chain.invoke({"text": doc})
            return summary.content if not isinstance(summary, str) else summary

        with concurrent.futures.ThreadPoolExecutor() as executor:
            chunk_summaries = list(executor.map(process_doc, split_docs))

        prompt = PromptTemplate(
            template="""You are provided with several summaries from different chunks of a document.
    Please merge them into a single, cohesive summary that captures the overall main themes. 

    Summaries:
    {summaries}

    """,
            input_variables=["summaries"]
        )
        chain = prompt | llm

        summaries_text = "\n".join(chunk_summaries)
        final_summary = chain.invoke({"summaries": summaries_text})

        print(f"Time taken to summarize the document: {time() - start}")

        self.paper_summary = final_summary.content if not isinstance(final_summary, str) else final_summary
            
    
    def get_topics(self):
        if self.paper_summary == "":
            self.summarize()
        prompt = PromptTemplate(
            template="""Here is it's summary: \n {summary} \n Get the scientific topics that are related to the abstarct above. Ouput only the keywords separated by a '/'. Desired format: Machine Learning/New Materials/NLP""", # Work on the prompt/output of LLM
            input_variables=["summary"]
        )
        chain = prompt | self.llm
        self.topics = chain.invoke({"summary": self.paper_summary}).content if isinstance(self.llm, ChatOpenAI) else chain.invoke({"summary": self.paper_summary})

    
    def __call__(self):
        """Run Spock."""
        self.download_pdf()
        self.add_custom_questions()
        self.scan_pdf() 
        
        if not self.paper_summary: 
            self.summarize()
            if not self.topics:
                self.get_topics()        
            
    
    def add_custom_questions(self): # Add custom metrics
        """Add custom questions to the questions dictionary."""
        
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
            temp_text = " Answer either 'Yes' or 'No' followed by a '/' then the exact sentence without any changes\
                                                from the document that supports your answer. If the answer is No or If you don't know the answer, say 'NA/None'" if self.settings['Binary Response'] else " Respond to the question followed by a '/' then the exact sentence without any changes\
                                                    from the document that supports your answer. If the answer is No or If you don't know the answer, say 'NA/None'"
            self.questions.update({question_topic.content if not isinstance(question_topic, str) else question_topic :{"question":question+temp_text, "output":{'response':"","sentence":""}}})
            
            
            
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
    
    
    
    def answer_question(self, question:str):
        """
        Answer a question
        """
        if self.vectorstore:
            return self.query_rag(question)
        else:
            self.chunk_indexing(self.paper)
            return self.query_rag(question)

