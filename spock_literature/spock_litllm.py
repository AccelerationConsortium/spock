from spock import Spock, verificator
from typing import List, Optional, Union
from spock_literature.texts import QUESTIONS, PAPERS_PATH
from pathlib import Path
import nvtx
from time import time



class Spock_litellm(Spock):
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
        super().__init__(
            model=model,
            paper=paper,
            custom_questions=custom_questions,
            publication_doi=publication_doi,
            publication_title=publication_title,
            publication_url=publication_url,
            papers_download_path=papers_download_path,
            temperature=temperature,
            embed_model=embed_model,
            folder_path=folder_path,
            settings=settings
        )        
        
    @nvtx.annotate("Summerize")
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
            return summary.content if hasattr(summary, "content") else summary

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

        self.paper_summary = final_summary.content if hasattr(final_summary, "content") else final_summary
