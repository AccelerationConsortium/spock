"""Main module."""
import json
import time
import concurrent.futures
from spock_literature.classes.Helper_LLM import Helper_LLM
from operator import itemgetter
import os
from langchain_community.document_loaders.pdf import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
import faiss
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import faiss
import os
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader, TextLoader
from langchain.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
import json
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
import requests
from scholarly import scholarly


class Spock(Helper_LLM): # Heritage a voir plus tard - maybe bot_llm
    """Spock class."""
    def __init__(self, paper, custom_questions:list=[]):
        """Initialize Spock."""
        super().__init__()
        self.paper =  paper # To edit later
        self.paper_summary = ""
        self.custom_questions = custom_questions
        self.questions = {
            "new materials":{"question":""""Does the document mention any new or novel materials discovered?\
                                            Examples sentences for new materials discovery:
                    1. Here we report the in vitro validation of eight novel GPCR peptide activators.
                    2. The result revealed that novel peptides accumulated only in adenocarcinoma lung cancer cell-derived xenograft tissue.
                    3. This led to the discovery of several novel catalyst compositions for ammonia decomposition, which were experimentally\
                        validated against "state-of-the-art" ammonia decomposition catalysts and were\
                        found to have exceptional low-temperature performance at substantially lower weight loadings of Ru.
                    4. We applied a workflow of combined in silico methods (virtual drug screening, molecular docking and supervised machine learning algorithms)\
                                            to identify novel drug candidates against COVID-19.\
                    Answer either 'Yes' or 'No' followed by a '/' then an exact sentence without any changes from the document that supports your answer. If the answer is No or If you don't know the answer, say 'NA/None'""" , "output": {'response':"","sentence":""}},
            
            
            "screening algorithms":{"question":"Does the document mention any new or novel high-throughput or large-scale screening algorithm, methods or workflow?\
                                            Examples sentences for new high-throughput screening algorithms:\
                                        1. In this study, we propose a large-scale (drug–target interactions) DTI prediction system,\
                                            DEEPScreen, for early stage drug discovery,\
                                            using deep convolutional neural network.\
                                        2. We performed a large-scale screening of fast-growing strains with 180 strains isolated\
                                            from 22 ponds located in a wide\
                                            geographic range from the tropics to cool-temperate.\
                                        3. In the present study, we leverage a recently developed high-throughput periodic\
                                            DFT workflow tailored for MOF structures\
                                            to construct a large-scale database of MOF quantum mechanical properties.\
                                            If there are any, what are the screening algorithms used in the document? Answer either\
                                            'Yes' or 'No' followed by a '/' then the exact sentence without any changes\
                                            from the document that supports your answer. If the answer is No or If you don't know the answer, say 'NA/None'", "output": {'response':"","sentence":""}},
            
            
            "experimental methodology":{"question":"Does the document mention any new or novel experimental methodology used?\
                                            Examples sentences for novel experimental methodology:\
                                        1. The current work presents a novel modified battery module configuration\
                                            employing two-layer nanoparticle enhanced phase change materials (nePCM)\
                                        2. A novel synthesis of new antibacterial nanostructures based on Zn-MOF\
                                            compound: design, characterization and a high performance application\
                                            If there are any, what are the screening algorithms used in the document? Answer either\
                                            'Yes' or 'No' followed by a '/' then the exact sentence without any changes\
                                        3. This study revealed the therapeutic potency of a novel hybrid peptide, \
                                        and supports the use of rational design in development of new antibacterial agents\
                                        'Yes' or 'No' followed by a '/' then the exact sentence without any changes\
                                            from the document that supports your answer. If the answer is No or If you don't know the answer, say 'NA/None'", "output": {'response':"","sentence":""}},
            
            "ML algirothms":{"question":"Does the document mention the development of any new machine learning and deep learning algorithm\
                                            or AI model/architecutre?\
                                        Examples sentences for new machine learning or deep learning algorithms :\
                                            1. In this study, we developed and optimized a novel and reliable hybrid machine learning\
                                            paradigm based on the extreme learning machines (ELM) method using the BAT algorithm optimization.\
                                        2. We here therefore introduce ionbot, a novel open modification search engine that is the first\
                                            to fully merge machine learning with peptide identification.\
                                        Answer either 'Yes' or 'No' followed by a '/' then the exact sentence without any changes from\
                                            the document that supports your answer. If the answer is No or If you don't know the answer, say 'NA/None'", "output": {'response':"","sentence":""}},
            
            "models":{"question":"DoAre specific new or novel methods, models and workflows used in the document?\
                                        Examples sentences for new methods and workflows :\
                                        1. We developed a novel synthesis method for hydrothermal reactions under a phosphoric acid medium\
                                            and obtained a series of metal polyiodates with strong SHG effects.\
                                        Answer either 'Yes' or 'No' followed by a '/' then\
                                            the exact sentence without any changes from the document that supports your answer. If the answer is No or If you don't know the answer, say 'NA/None'", "output": {'response':"","sentence":""}},
            "funding":{"question":"Does the document mention funding, award or financial support in the acknowledgements?\
                                        Examples sentences for funding or financial support:\
                                        1. This work is supported in part by the National Science Foundation under Award No. OIA-1946391.\
                                            Answer either 'Yes' or 'No' followed by a '/' then the exact sentence\
                                            without any changes from the document that supports your answer. If the answer is No or If you don't know the answer, say 'NA/None'", "output": {'response':"","sentence":""}},
        
        "material datasets":{"question":"Does the document mention the development of any new material-related datasets or databases? \
                                        Examples sentences for new dataset/database:\
                                        1. In this study, we thus designed a database of ∼20,000 hypothetical MOFs, keeping in mind their \
                                        chemical diversity in terms of pore geometry, metal chemistry, linker chemistry, and functional groups\
                                        2. In the present study, we leverage a recently developed high-throughput periodic DFT\
                                            workflow tailored for MOF structures\
                                            to construct a large-scale database of MOF quantum mechanical properties. \
                                        3. In this work, to build the predictive tool, a dataset was constructed\
                                            and models were trained and tested at a ratio of 75:25.\
                                            Answer either 'Yes' or 'No' followed by a '/' then the exact sentence without any changes from the document\
                                            that supports your answer. If the answer is No or If you don't know the answer, say 'NA/None'", "output": {'response':"","sentence":""}},
        
        "drug formulations explored":{"question":"Has the document mentioned exploring any drug formulations?\
                                        Examples sentences for exploring drug formulations:\
                                        1. In this review, we tried to explore the major considerations and target factors in\
                                            drug delivery through the nasal-brain\
                                            route based on physiological knowledge and formulation research information.\
                                        2. Notably, they can be incorporated in pharmaceutical formulations to enhance drug solubility,\
                                            absorption, and bioavailability due to the formulation itself and the P-gp inhibitory effects of the excipients.\
                                            Answer either 'Yes' or 'No' followed by a '/' then the exact sentence without any changes from the document\
                                            that supports your answer. If the answer is No or If you don't know the answer, say 'NA/None'", "output": {'response':"","sentence":""}},
        
        "lead small-molecule drug candidates":{"question":"Does the document mention identifying to developing lead small-molecule drug candidates?\
                                        Example sentences for new lead small-molecule drug candidates:\
                                        1. Discovery of a Small Molecule Drug Candidate for Selective NKCC1 Inhibition in Brain Disorders\
                                        2. We have setup a drug discovery program of small-molecule compounds that act as chaperones enhancing\
                                            TTR/Amyloid-beta peptide (Aβ) interactions.\
                                            Answer either 'Yes' or 'No' followed by a '/' then the exact sentence without any changes from\
                                            the document that supports your answer. If the answer is No or If you don't know the answer, say 'NA/None'", "output": {'response':"","sentence":""}},
        
        
        
        "clinical trials":{"question":"Are there any clinical trials mentioned in the document?\
                                        Example sentences for clinical trials:\
                                        1. Here, we provide recommendations from the BEAT-HIV Martin Delaney Collaboratory\
                                            on which viral measurements should be prioritized in HIV-cure-directed clinical trials\
                                        2.  This 12-month longitudinal, 2-group randomized clinical trial recruited MSM through\
                                            online banner advertisements from March through August 2015.\
                                        3. Efficacy of hydroxychloroquine in patients with COVID-19: results of a randomized clinical trial\
                                            Answer either 'Yes' or 'No' followed by a '/' then the exact sentence without any changes\
                                                from the document that supports your answer. If the answer is No or If you don't know the answer, say 'NA/None'", "output": {'response':"","sentence":""}}}

        
    
    
    def download_pdf(self):
        """Download the PDF of a publication."""
        pass
    
    
    
        #if self.paper isinstance Publication_doi: # A voir/ Travailler sur la condition et OOP
            #self.pdf = self.paper.get_pdf()
        
        # We assume that the variable paper is a Publication object that contains the DOI of the publication
        
        
    
    def scan_pdf(self):
        """Scan the PDF of a publication."""
        
        self.chunk_indexing(self.paper)
        for question in self.questions:
            try:
                temp_response = self.query_rag(self.questions[question]['question']).split("/")
            except:
                temp_response = ["NA","None"]
            self.questions[question]['output']['response'] = temp_response[0]
            self.questions[question]['output']['sentence'] = temp_response[1]
        
        
        
        
        
    def summarize(self):
        """Return the summary of the publication."""
        temp_llm = Ollama(model="llama3.1", temperature=0.05)
        prompt = PromptTemplate(
            template="You are an AI assistant that is going to help me summarize documents. Here is one page of the document, can you summarize the page please. Output only the summary {document}",
            input_variables=["document"]
        )
        # Summarizing each page then merging everything
        pages = PyPDFLoader(self.paper).load()
        print(len(pages))
        summaries = ""
        for i,page in enumerate(pages):
            chain = prompt | temp_llm
            summaries += f"Summary of page {i+1} {chain.invoke({'document': page})}"
            
        prompt = PromptTemplate(
            template="You are an AI assistant that is going to help me summarize documents. Since the document is really long, we summerized each and every page, and your task is to merge all of these summaries to give me a new global summary. Here are summaries {document} \n \n Output only the summary.", # Maybe allow user to choose the length of the summary
            input_variables=["document"]
        )
        chain = prompt | temp_llm
        self.paper_summary = chain.invoke({"document": summaries})

            
    def get_topics(self):
        if self.paper_summary == "":
            self.summarize()
        prompt = PromptTemplate(
            template="You are an AI assistant, and here is a document. get the topic of the document. Here is it's summary: \n {summary} \n Get the scientific topics that are related to the abstarct above. Ouput only the keywords separated by a '/'. Example: Machine Learning/New Materials", # Work on the prompt/output of LLM
            input_variables=["summary"]
        )
        chain = prompt | self.llm
        
        #TODO: Continue from here/Add topics
        return chain.invoke({"summary": self.paper_summary})
        
        
        
    
    def __call__(self):
        """Run Spock."""
        self.download_pdf()
        print("Downloaded the PDF")
        #self.add_custom_questions()
        self.scan_pdf()
        print("Scanned the PDF")
        self.summarize()
        print("Summarized the PDF")
        topics = self.get_topics()
        print("Got the topics")
        # Format the output text
        output_text = f"Here is a summary of the publication: \n {self.paper_summary}\n  ---- \nHere are the topics of the publication: {topics}\n ---"
        for question in self.questions:
            output_text += f"Question: {question}\n"
            output_text += f"Answer: {self.questions[question]['output']['response']}\n"
            output_text += f"Supporting sentence: {self.questions[question]['output']['sentence']}\n"
            output_text += "\n --- \n"
            
        return output_text
        
        
    
    def add_custom_questions(self):
        """Add custom questions to the questions dictionary."""
        
        
        for question in self.custom_questions:
            temp_llm = Ollama(model="llama3.1", temperature=0.05)
            prompt = PromptTemplate(
                template="""Here is a question, I want you to give me to what topic it is related the most {question}. \
                Here are some examples to help you: \
                
                Question: Does the document mention any new or novel materials discovered?\
                                        Examples sentences for new materials discovery:
                1. Here we report the in vitro validation of eight novel GPCR peptide activators.
                2. The result revealed that novel peptides accumulated only in adenocarcinoma lung cancer cell-derived xenograft tissue.
                3. This led to the discovery of several novel catalyst compositions for ammonia decomposition, which were experimentally\
                    validated against "state-of-the-art" ammonia decomposition catalysts and were\
                    found to have exceptional low-temperature performance at substantially lower weight loadings of Ru.
                4. We applied a workflow of combined in silico methods (virtual drug screening, molecular docking and supervised machine learning algorithms)\
                                        to identify novel drug candidates against COVID-19.\
                Output: new materials\
                    
                -- \
                Does the document mention any new or novel high-throughput or large-scale screening algorithm, methods or workflow?\
                                        Examples sentences for new high-throughput screening algorithms:\
                                       1. In this study, we propose a large-scale (drug–target interactions) DTI prediction system,\
                                        DEEPScreen, for early stage drug discovery,\
                                        using deep convolutional neural network.\
                                       2. We performed a large-scale screening of fast-growing strains with 180 strains isolated\
                                        from 22 ponds located in a wide\
                                        geographic range from the tropics to cool-temperate.\
                                       3. In the present study, we leverage a recently developed high-throughput periodic\
                                        DFT workflow tailored for MOF structures\
                                        to construct a large-scale database of MOF quantum mechanical properties.\
                                        If there are any, what are the screening algorithms used in the document? Answer either\
                                        'Yes' or 'No' followed by a '/' then the exact sentence without any changes\
                                            
                Output: screening algorithms\
                    
                Youe output should only contain the topic of the question.\
                
                
                """,
                input_variables=["question"]
            )
            
            chain = prompt | temp_llm
            question_topic = chain.invoke({"question":question})
            self.questions.update({question_topic:{"question":question , "output":{'response':"","sentence":""}}})
            
            
            
            


if __name__ == "__main__": # That would be the script to submit for the job
    import sys
    #spock = Spock(paper=sys.argv[1], custom_questions=sys.argv[2:]) # Update - to see layer how to fix it 
