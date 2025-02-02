"""Main module."""
from langchain_openai import ChatOpenAI
from spock_literature.utils.Helper_LLM import Helper_LLM
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from spock_literature.texts import get_questions, PAPERS_PATH
from langchain_ollama import OllamaLLM
from spock_literature.utils.Generate_podcast import generate_audio
from pathlib import Path
from typing import List, Optional, Union
from spock_literature.utils.Url_downloader import URLDownloader
from langchain.schema import Document
from scidownl import scihub_download

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
        super().__init__(model=model, temperature=temperature, embed_model=embed_model, folder_path=folder_path)
        self.paper: Optional[Path] = Path(paper) if paper else None
        self.paper_summary: str = ""
        self.custom_questions: List[str] = custom_questions or []
        self.publication_doi: Optional[str] = publication_doi
        self.publication_title: Optional[str] = publication_title
        self.publication_url: Optional[str] = publication_url
        self.topics: str = ""
        self.settings = settings
        self.questions = get_questions(self.settings['Binary Response'])
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
        
        self.chunk_indexing(self.paper)
        lim = 0 if self.settings['Questions'] else 10
        for i,question in enumerate(self.questions):
            if i >= lim:
                try:
                    temp_response = self.query_rag(self.questions[question]['question'])
                except Exception as e:
                    print("An error occured while scanning the PDF for the question: ", question)
                    temp_response = "NA/None"
                if self.settings['Binary Response']:
                    temp_response = temp_response.split('/')
                    self.questions[question]['output']['response'] = temp_response[0]
                    self.questions[question]['output']['sentence'] = temp_response[1]
                else:
                    
                    ##### To update
                    try:
                        temp_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
                        prompt = PromptTemplate(
                            template="Here is a text {text}. It contains an answer followed by some extracts from a text supporting that answer. The output should look like this: Answer/Supporting answers. If the answer is 'no' or there is no Supporting sentence mentioned in the text, output, followed by a '/None'",
                            input_variables=["text"]
                        )
                        chain = prompt | temp_llm
                        
                        # To verify le .content si c'est chatgpt ou llama
                        temp_response = chain.invoke({"text":temp_response}).content.split('/')
                        self.questions[question]['output']['response'] = temp_response[0]
                        self.questions[question]['output']['sentence'] = temp_response[1]
                    except Exception as e:
                        print("An error occured while scanning the PDF for the question: ", question)
                        self.questions[question]['output']['response'] = "NA"
                        self.questions[question]['output']['sentence'] = "None"
                        
                    ######
    
    
    def summarize(self) -> None:
        from langchain.docstore.document import Document
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.chains.llm import LLMChain
        from langchain.prompts import PromptTemplate
        from langchain_openai import ChatOpenAI
        from time import time
        import concurrent.futures

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
            
    
    
    def get_topics(self) -> None:
        """Get the topics covered in the publication."""
        pass
    
    
    
    
    
    def __call__(self):
        """Run Spock."""
        self.download_pdf()
        self.add_custom_questions()
        self.scan_pdf() 
        
        if self.settings['Summary']:
            if not self.paper_summary: 
                self.summarize()
            if not self.topics:
                self.get_topics()        
        
        
    
    def add_custom_questions(self):
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
    
    def verificator(self):
        """
        Verify if input is good
        """
        pass
    
    
    def answer_question(self, question:str):
        """
        Answer a question
        """
        pass




if __name__ == "__main__":
    spock = Spock(
        model="gpt-4o",
        paper="data-sample.pdf",
    )
    spock.summarize()
    print(spock.paper_summary)
