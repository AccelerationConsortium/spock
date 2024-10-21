"""Main module."""
import json
import time
import concurrent.futures
from classes.Helper_LLM import Helper_LLM


class Spock(Helper_LLM): # Heritage a voir plus tard - maybe bot_llm
    """Spock class."""
    def __init__(self, paper, custom_questions:list=[]):
        """Initialize Spock."""
        super().__init__()
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
        for question in self.questions:
            temp_response = self.query_rag(self.questions[question]['question']).split("/")
            self.questions[question]['output']['response'] = temp_response[0]
            self.questions[question]['output']['sentence'] = temp_response[1]
        
        
        
        
        
          
    
    def __call__(self):
        """Run Spock."""
        self.download_pdf()
        self.add_custom_questions()
        self.scan_pdf()
        
        # Format the output text
        
        output_text = ""
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
            
            
            
            
if __name__ == "__main__":
    spock = Spock(paper="test")
    #spock()
    