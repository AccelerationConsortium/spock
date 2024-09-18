from LLM import LLM
import os
import json
from utilities import Bot_LLM

with open("llm_ouput.json", "r") as f:
    response = json.load(f)
    
    
    
pdf_list = os.listdir("papers/papers")
#print(pdf_list)
#response = {}
format_instruction = "Answer  either 'Yes' or 'No' followed by a '/' then an exact sentence from the document that\
      supports your answer. If you don't know the answer, say 'NA/'"
for i in range(10):
    #name = pdf_list[i].split('.')[-1].split('/')[-1].replace("_","/")
    
    name = pdf_list[i]
    print(name)
    if not os.path.exists(f'files_output/{name}'):
        try:
            response[name] = {}
            llm = Bot_LLM(model="llama3.1", folder_path='db/db'+str(i))
            llm.chunk_indexing("papers/papers/"+pdf_list[i])    


            
            #llm.set_folder_path("db/db"+str(i))
            
            
            print("running chunk_indexing")
        # llm.split_and_embedding_chunk_pdf("papers/papers/"+pdf_list[i])
            # Process 'affiliation'
        # output_llm = llm.query_rag("What are the authors affiliation. Output a dictionary ? Output either 'Yes' or 'No' followed by a '/' then a sentence from the document that supports your answer.")
        # response[i]['affiliation'] = {'Yes/No': output_llm.split('/')[0].strip(), 'sentence': output_llm.split('/')[1]}
        # print("-----")

            # Process 'topic'
        # output_llm = llm.query_rag("What are topics of the paper? Output the topic of the paper '/' then a sentence from the document that supports your answer.") # To updatee
        # response[i]['topic'] =  output_llm.split('/')[0].strip()
            #response[i]['topic'].update({'sentence': output_llm.split('/')[1]})
            #print("-----")

            # Process 'New composition of matter discovered'
            
            output_llm = llm.query_rag("""Does the document mention any new or novel materials discovered?\
                                        Examples sentences for new materials discovery:
                1. Here we report the in vitro validation of eight novel GPCR peptide activators.
                2. The result revealed that novel peptides accumulated only in adenocarcinoma lung cancer cell-derived xenograft tissue.
                3. This led to the discovery of several novel catalyst compositions for ammonia decomposition, which were experimentally\
                    validated against "state-of-the-art" ammonia decomposition catalysts and were\
                    found to have exceptional low-temperature performance at substantially lower weight loadings of Ru.
                4. We applied a workflow of combined in silico methods (virtual drug screening, molecular docking and supervised machine learning algorithms)\
                                        to identify novel drug candidates against COVID-19.\
                Answer either 'Yes' or 'No' followed by a '/' then an exact sentence without any changes from the document that supports your answer.""")
            response[name]['new materials'] = {'Yes/No': output_llm.split('/')[0].strip(), 'sentence': output_llm.split('/')[1]}
            print("-----")

            # Process 'Large-scale screening algorithms, methods descriptions, workflows, models'
            output_llm = llm.query_rag("Does the document mention any new or novel high-throughput or large-scale screening algorithm, methods or workflow?\
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
                                        from the document that supports your answer.")
            response[name]['screening algorithms'] = {'Yes/No': output_llm.split('/')[0].strip(), 'sentence': output_llm.split('/')[1]}
            print("-----")

            # Process 'Experimental methodology'
            output_llm = llm.query_rag("Does the document mention any new or novel experimental methodology used?\
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
                                        from the document that supports your answer.")
            response[name]['experimental methodology'] = {'Yes/No': output_llm.split('/')[0].strip(), 'sentence': output_llm.split('/')[1]}
            print("-----")

            # Process 'AI algorithms' To edit
            output_llm = llm.query_rag("""Does the document mention the development of any new machine learning and deep learning algorithm\
                                        or AI model/architecutre?\
                                       Examples sentences for new machine learning or deep learning algorithms :\
                                        1. In this study, we developed and optimized a novel and reliable hybrid machine learning\
                                        paradigm based on the extreme learning machines (ELM) method using the BAT algorithm optimization.\
                                       2. We here therefore introduce ionbot, a novel open modification search engine that is the first\
                                        to fully merge machine learning with peptide identification.\
                                     Answer either 'Yes' or 'No' followed by a '/' then the exact sentence without any changes from\
                                        the document that supports your answer.""")
            response[name]['ML algorithms'] = {'Yes/No': output_llm.split('/')[0].strip(), 'sentence': output_llm.split('/')[1]}
            print("-----")

            # Process 'Novel methods and models'
            output_llm = llm.query_rag("Are specific new or novel methods, models and workflows used in the document?\
                                       Examples sentences for new methods and workflows :\
                                       1. We developed a novel synthesis method for hydrothermal reactions under a phosphoric acid medium\
                                        and obtained a series of metal polyiodates with strong SHG effects.\
                                       Answer either 'Yes' or 'No' followed by a '/' then\
                                        the exact sentence without any changes from the document that supports your answer.")
            response[name]['models'] = {'Yes/No': output_llm.split('/')[0].strip(), 'sentence': output_llm.split('/')[1]}
            print("-----")

            # Process 'funding' 
            # Add more context to the question/ check acknowlegdment section
            output_llm = llm.query_rag("Does the document mention funding, award or financial support in the acknowledgements?\
                                       Examples sentences for funding or financial support:\
                                       1. This work is supported in part by the National Science Foundation under Award No. OIA-1946391.\
                                        Answer either 'Yes' or 'No' followed by a '/' then the exact sentence\
                                        without any changes from the document that supports your answer.")
            response[name]['funding'] = {'Yes/No': output_llm.split('/')[0].strip(), 'sentence': output_llm.split('/')[1]}
            print("-----")

            # Process 'material datasets'
            output_llm = llm.query_rag("Does the document mention the development of any new material-related datasets or databases? \
                                       Examples sentences for new dataset/database:\
                                       1. In this study, we thus designed a database of ∼20,000 hypothetical MOFs, keeping in mind their \
                                       chemical diversity in terms of pore geometry, metal chemistry, linker chemistry, and functional groups\
                                       2. In the present study, we leverage a recently developed high-throughput periodic DFT\
                                        workflow tailored for MOF structures\
                                        to construct a large-scale database of MOF quantum mechanical properties. \
                                       3. In this work, to build the predictive tool, a dataset was constructed\
                                        and models were trained and tested at a ratio of 75:25.\
                                        Answer either 'Yes' or 'No' followed by a '/' then the exact sentence without any changes from the document\
                                        that supports your answer.")
            response[name]['material datasets'] = {'Yes/No': output_llm.split('/')[0].strip(), 'sentence': output_llm.split('/')[1]}
            print("-----")

            # Process 'drug formulations explored'
            output_llm = llm.query_rag("Has the document mentioned exploring any drug formulations?\
                                       Examples sentences for exploring drug formulations:\
                                       1. In this review, we tried to explore the major considerations and target factors in\
                                        drug delivery through the nasal-brain\
                                        route based on physiological knowledge and formulation research information.\
                                       2. Notably, they can be incorporated in pharmaceutical formulations to enhance drug solubility,\
                                        absorption, and bioavailability due to the formulation itself and the P-gp inhibitory effects of the excipients.\
                                        Answer either 'Yes' or 'No' followed by a '/' then the exact sentence without any changes from the document\
                                        that supports your answer.")
            response[name]['drug formulations explored'] = {'Yes/No': output_llm.split('/')[0].strip(), 'sentence': output_llm.split('/')[1]}
            print("-----")

            # Process 'novel drug formulations'
            output_llm = llm.query_rag("Does the document identify any novel drug formulations? \
                                       Examples sentences for identifying novel drug formulations:\
                                       1. We applied a workflow of combined in silico methods (virtual drug screening,\
                                        molecular docking and supervised machine learning algorithms)\
                                          to identify novel drug candidates against COVID-19\
                                       2. Therefore, we designed a novel formulation KK-46 based on peptide dendrimers (PD) to achieve safe\
                                        and efficient siRNA delivery into the lung \
                                        Answer either 'Yes' or 'No' followed by a '/' then the exact sentence without any changes\
                                        from the document that supports your answer.")
            response[name]['novel drug formulations'] = {'Yes/No': output_llm.split('/')[0].strip(), 'sentence': output_llm.split('/')[1]}
            print("-----")

            # Process 'lead small-molecule drug candidates'
            output_llm = llm.query_rag("Does the document mention identifying to developing lead small-molecule drug candidates?\
                                       Example sentences for new lead small-molecule drug candidates:\
                                       1. Discovery of a Small Molecule Drug Candidate for Selective NKCC1 Inhibition in Brain Disorders\
                                       2. We have setup a drug discovery program of small-molecule compounds that act as chaperones enhancing\
                                        TTR/Amyloid-beta peptide (Aβ) interactions.\
                                        Answer either 'Yes' or 'No' followed by a '/' then the exact sentence without any changes from\
                                        the document that supports your answer.")
            response[name]['lead small-molecule drug candidates'] = {'Yes/No': output_llm.split('/')[0].strip(), 'sentence': output_llm.split('/')[1]}
            print("-----")

            # Process 'clinical trials'
            output_llm = llm.query_rag("Are there any clinical trials mentioned in the document?\
                                       Example sentences for clinical trials:\
                                       1. Here, we provide recommendations from the BEAT-HIV Martin Delaney Collaboratory\
                                        on which viral measurements should be prioritized in HIV-cure-directed clinical trials\
                                       2.  This 12-month longitudinal, 2-group randomized clinical trial recruited MSM through\
                                        online banner advertisements from March through August 2015.\
                                       3. Efficacy of hydroxychloroquine in patients with COVID-19: results of a randomized clinical trial\
                                        Answer either 'Yes' or 'No' followed by a '/' then the exact sentence without any changes\
                                        from the document that supports your answer.")
            response[name]['clinical trials'] = {'Yes/No': output_llm.split('/')[0].strip(), 'sentence': output_llm.split('/')[1]}
            print("-----")
             
            with open(f"files_output/{name}.json", "w") as f:
                json.dump(response, f, indent=4)
        except Exception as e:
            print(e)
            continue


print(response)

