
import os
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader, TextLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.llms import Ollama
import json
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
import requests
from scholarly import scholarly

class Publication:
    def __init__(self,publication_filled, llm_use:bool=True) -> None:
        self.publication_filled = publication_filled
        self.title = self.get_publication_title()
        self.abstract = self.get_publication_abstract().lower()
        self.author = self.get_author_name()
        self.year = self.get_year()
        self.url = self.get_publication_url()
        self.citation = self.get_citation()
        self.pdf = self.get_pdf()
        self.topic = None
    
      
      
    '''  
    def get_topic(self,output_file="json/ouput.json", # à voir cette histoire avec get_topic et __get_topic
                  input_file="json/response.json") -> list[str]:
        try:
            with open(output_file,'r') as file:
                data = json.load(file)
            return data[self.author]['topic']
        except Exception as e:
            return self.__get_topic(input_file)
     '''   
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
    
    def get_topic(self,llm) -> None:
        self.topic: dict = llm.get_topic_publication()
        return self.topic
    
    def get_pdf(self):
        query = self.title.lower().replace(" ", "+")
        url = f"https://scholar.google.com/scholar?q={query}"
        response = requests.get(url)
        if response.status_code == 200:
            html_content = response.text
            try:
                return self.__parse_google_scholar(html_content)
            except Exception as e:
                print(f"An error occurred while fetching the PDF link: {e}")
        else:
            print(f"Failed to fetch the page. Status code: {response.status_code}")
            return None

    def __repr__(self) -> str:
        self.available_attributes = {'title': self.title if self.title is not None else 'N/A',
                                     'abstract' : self.abstract if self.abstract is not None else 'N/A',
                                        'author': self.author if self.author is not None else 'N/A',
                                        'year': self.year if self.year is not None else 'N/A',
                                        'url': self.url if self.url is not None else 'N/A',
                                        'citation': self.citation if self.citation is not None else 'N/A',
                                        'pdf': self.pdf if self.pdf is not None else 'N/A',
                                        'topic': self.topic if self.topic is not None else 'N/A'}
        return str(self.available_attributes)


        
    def __parse_google_scholar(self,html_content):

        soup = BeautifulSoup(html_content, 'html.parser')

        a_tags = soup.find_all('a')
        try:
            pdf_link = [a['href'] for a in a_tags if 'href' in a.attrs and '.pdf' in a['href']][0]
            print(f"PDF link found: {pdf_link}")
            self.download_pdf(link=pdf_link)
            return pdf_link
        except Exception as e:
            print(f"An error occurred while parsing the PDF link: {e}")
        
        
    
    def download_pdf(self,path=os.getcwd()+"/pdfs/",link=""):
        
        # Verifier si le path exist sinon le creer
        
        import os
        import requests
                
        
        path = path + self.title + ".pdf"
        if link != "":
            try:
                response = requests.get(link)
                if response.status_code == 200:
                    with open(path, 'wb') as file:
                        file.write(response.content)
                    print(f"PDF successfully downloaded and saved to {path}")
                else:
                    print(f"Failed to download the PDF. HTTP Status Code: {response.status_code}")

            except requests.exceptions.RequestException as e:
                print(f"An error occurred while downloading the PDF: {e}")
        else:
            from scidownl import scihub_download
            #  scidownl by title
            try:
                scihub_download(self.title, paper_type="title", out=path)
            except:
                try:
                    # By URL
                    scihub_download(self.pdf, out=path)
                except:
                    print("Couldn't download the PDF")
           

        
        
    
     
                    
            
        

from operator import itemgetter
import os
from langchain_community.document_loaders.pdf import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma


class Bot_LLM:
    def __init__(self,model='llama3',embed_model='mxbai-embed-large', folder_path='db2'):
        self.llm = Ollama(model=model)
        self.oembed = OllamaEmbeddings(model=embed_model)
        self.folder_path = folder_path
        self.vectorestore = None

    
    
    def rag_fusion(self, question):
        from langchain.prompts import ChatPromptTemplate

        # RAG-Fusion: Related
        template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
        Generate multiple search queries related to: {question} \n
        Output (4 queries):"""
        
        
        prompt_rag_fusion = ChatPromptTemplate.from_template(template)
        from langchain_core.output_parsers import StrOutputParser

        generate_queries = (
            prompt_rag_fusion 
            | self.llm(temperature=0)
            | StrOutputParser() 
            | (lambda x: x.split("\n"))
        )
        from langchain.load import dumps, loads

        def reciprocal_rank_fusion(results: list[list], k=60):
            """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
                and an optional parameter k used in the RRF formula """
            
            # Initialize a dictionary to hold fused scores for each unique document
            fused_scores = {}

            # Iterate through each list of ranked documents
            for docs in results:
                # Iterate through each document in the list, with its rank (position in the list)
                for rank, doc in enumerate(docs):
                    # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
                    doc_str = dumps(doc)
                    # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
                    if doc_str not in fused_scores:
                        fused_scores[doc_str] = 0
                    # Retrieve the current score of the document, if any
                    previous_score = fused_scores[doc_str]
                    # Update the score of the document using the RRF formula: 1 / (rank + k)
                    fused_scores[doc_str] += 1 / (rank + k)

            # Sort the documents based on their fused scores in descending order to get the final reranked results
            reranked_results = [
                (loads(doc), score)
                for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
            ]

            # Return the reranked results as a list of tuples, each containing the document and its fused score
            return reranked_results

        retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
        docs = retrieval_chain_rag_fusion.invoke({"question": question})
        len(docs)
        from langchain_core.runnables import RunnablePassthrough

        # RAG
        template = """Answer the following question based on this context:

        {context}

        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)

        final_rag_chain = (
            {"context": retrieval_chain_rag_fusion, 
            "question": itemgetter("question")} 
            | prompt
            | self.llm
            | StrOutputParser()
        )

        final_rag_chain.invoke({"question":question})


        
        
        

    def get_topic_publication(self):
        
        parser = JsonOutputParser()
        
        new_text = """
        {
            "topic": {
                "Machine Learning": ["Keyword1", "Keyword2", "Keyword3"],
                "Batteries": ["Keyword1", "Keyword2", "Keyword3"]
            }
        }
        """

        prompt = PromptTemplate(
            template="Please identify the topics from the following list the text given to you. Note: A single text can belong to multiple topics, so please list all relevant topics. {format_instructions}",
            input_variables=["format_instructions"]
        )

        chain = prompt | self.llm | parser
        topics = chain.invoke({"format_instructions": new_text})
        return topics['topic']

    
    def chunk_indexing(self, document:str):
        # Check if the document is a valid file path
        from langchain_experimental.text_splitter import SemanticChunker


        data = []
        if isinstance(document, str) and os.path.isfile(document):
            try:
                loader = PDFPlumberLoader(document)
                data = loader.load()
                chunk_size = 2000
                chunk_overlap = 20

            except Exception as e:
                raise RuntimeError(f"Error loading PDF: {e}")
        else:
            try:
                # Treat the document as raw text content
                if isinstance(document, str):
                    data.append(Document(page_content=document))
                elif isinstance(document, list):
                    for text in document:
                        data.append(Document(page_content=text))
                chunk_size = 180
                chunk_overlap = 5

            except Exception as e:
                raise RuntimeError(f"Error processing text: {e}")

            
        text_splitter=text_splitter = SemanticChunker(
    self.oembed, breakpoint_threshold_type="standard_deviation")


        all_splits = text_splitter.split_documents(data)
        self.vectorstore = Chroma.from_documents(documents=all_splits, embedding=self.oembed, persist_directory=self.folder_path)
        
    def query_rag(self, question:str) -> None:
        if self.vectorstore:
            docs = self.vectorstore.similarity_search(question)
            from langchain.chains import RetrievalQA
            qachain=RetrievalQA.from_chain_type(self.llm, retriever=self.vectorstore.as_retriever(), verbose=True)
            res = qachain.invoke({"query": question})
            print(res['result'])
            return res['result']


        else:
            raise Exception("No documents loaded")


class Author:
    def __init__(self, author):
        """
        Initialize an Author object.

        Args:
            author (str): The name of the author.
        """
        self.author_name = author
        
    def __repr__(self):
        """
        Return a string representation of the Author object.

        Returns:
            str: The name of the author.
        """
        return self.author_name

    def get_last_publication(self):
        """
        Get the last publication of the author.

        Returns:
            dict: A dict containing information about the last publication.
        """
        search_query = scholarly.search_author(self.author_name)
        first_author_result = next(search_query)
        author = scholarly.fill(first_author_result)
        first_publication = sorted(author['publications'], 
                                   key=lambda x: int(x['bib']['pub_year'])
                                   if 'pub_year' in x['bib'] else 0, 
                                   reverse=True)[0]
        first_publication_filled = scholarly.fill(first_publication)
        return first_publication_filled

    def setup_author(self, output_file,publication:Publication=None):
        """
        Setup the author by adding their last publication to a JSON file.

        Args:
            output_file (str): The path to the JSON file.

        Returns:
            None
        """
        with open(output_file, 'r') as file:
            data = json.load(file)
        author_last_publication = Publication(self.get_last_publication()) if publication != None else publication
        

        
        data[self.author_name] = {
            "title": author_last_publication.title,
            "abstract": author_last_publication.abstract,
            "author": author_last_publication.author, 
            "year": author_last_publication.year,
            "url": author_last_publication.url,
            "pdf": author_last_publication.pdf,
        }
        
        
        with open(output_file, 'w') as file:
            json.dump(data, file)

'''
author = Author('Mehrad Ansari')
pub = Publication(author.get_last_publication())

pub.download_pdf('pdfs/')
'''
