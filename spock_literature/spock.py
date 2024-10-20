"""Main module."""
import json
import time
import concurrent.futures
from publication import Publication
from author import Author
from spock_literature.classes.Helper_LLM import Helper_LLM


class Spock(Helper_LLM): # Heritage a voir plus tard - maybe bot_llm
    """Spock class."""
    def __init__(self, custom_questions:list=[], file_path:str=""):
        """Initialize Spock."""
        super().__init__()
        self.fielpath = file_path
        self.custom_questions = custom_questions
        self.questions = {
            "":{"question":"" ,
                "output":
                    {'response':"","sentence":""}}
            
        }

        
    
    
    def download_pdf(self):
        """Download the PDF of a publication."""
        pass
    
    def scan_pdf(self):
        """Scan the PDF of a publication."""
          
    
    def __call__(self):
        """Run Spock."""
        self.add_custom_questions()
        self.scan_pdf()
        
    
    def add_custom_questions(self):
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
                    
                Output only the topic of the question.\
                
                
                """,
                input_variables=["question"]
            )
            
            chain = prompt | temp_llm
            question_topic = chain.invoke({"question":question})
            self.questions.update({question_topic:{"question":question , "output":{'response':"","sentence":""}}})
            