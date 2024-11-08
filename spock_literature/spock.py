"""Main module."""
from spock_literature.classes.Helper_LLM import Helper_LLM
from operator import itemgetter
import os
from langchain.chains.combine_documents import create_stuff_documents_chain
import faiss
from langchain_community.document_loaders import PyPDFLoader
import faiss
import os
import json
from langchain_core.prompts import PromptTemplate
import requests
from scholarly import scholarly
from spock_literature.classes.docs import LoadDoc
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
            
            "models":{"question":"Are there specific new or novel methods, models and workflows used in the document?\
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
            token_max=4000,
        )

        map_reduce_chain = MapReduceDocumentsChain(
            llm_chain=map_chain,
            reduce_documents_chain=reduce_documents_chain,
            document_variable_name="docs",
            return_intermediate_steps=False,
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1750, chunk_overlap=20
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
