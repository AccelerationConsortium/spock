from langchain_community.vectorstores import FAISS
import faiss


import os
import requests
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
#from  ../spock_
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
def to_run():
    #from LLM import LLM
    import os
    import json
    from utilities import Bot_LLM
    response = {}
    #with open("llm_ouput.json", "r") as f:
        #response = json.load(f)
        
        
        
    pdf_list = os.listdir("papers/")
    #print(pdf_list)
    #response = {}
    format_instruction = "Answer  either 'Yes' or 'No' followed by a '/' then an exact sentence from the document that\
        justifies your answer. If the answer is No or If you don't know the answer, say 'NA/None'"
    for i in range(1):
        #name = pdf_list[i].split('.')[-1].split('/')[-1].replace("_","/")
        
        name = pdf_list[i]
        print(name)
        if not os.path.exists(f'files_output/{name}'):
            try:
                response = {}
                response[name] = {}
                llm = Bot_LLM(model="llama3.1:70b", folder_path='db/db'+str(i))
                llm.chunk_indexing("papers/"+pdf_list[i])    


                
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
                    Answer either 'Yes' or 'No' followed by a '/' then an exact sentence without any changes from the document that supports your answer. If the answer is No or If you don't know the answer, say 'NA/None'""")
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
                                            from the document that supports your answer. If the answer is No or If you don't know the answer, say 'NA/None'")
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
                                            from the document that supports your answer. If the answer is No or If you don't know the answer, say 'NA/None'")
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
                                            the document that supports your answer. If the answer is No or If you don't know the answer, say 'NA/None'""")
                response[name]['ML algorithms'] = {'Yes/No': output_llm.split('/')[0].strip(), 'sentence': output_llm.split('/')[1]}
                print("-----")

                # Process 'Novel methods and models'
                output_llm = llm.query_rag("Are specific new or novel methods, models and workflows used in the document?\
                                        Examples sentences for new methods and workflows :\
                                        1. We developed a novel synthesis method for hydrothermal reactions under a phosphoric acid medium\
                                            and obtained a series of metal polyiodates with strong SHG effects.\
                                        Answer either 'Yes' or 'No' followed by a '/' then\
                                            the exact sentence without any changes from the document that supports your answer. If the answer is No or If you don't know the answer, say 'NA/None'")
                response[name]['models'] = {'Yes/No': output_llm.split('/')[0].strip(), 'sentence': output_llm.split('/')[1]}
                print("-----")

                # Process 'funding' 
                # Add more context to the question/ check acknowlegdment section
                output_llm = llm.query_rag("Does the document mention funding, award or financial support in the acknowledgements?\
                                        Examples sentences for funding or financial support:\
                                        1. This work is supported in part by the National Science Foundation under Award No. OIA-1946391.\
                                            Answer either 'Yes' or 'No' followed by a '/' then the exact sentence\
                                            without any changes from the document that supports your answer. If the answer is No or If you don't know the answer, say 'NA/None'")
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
                                            that supports your answer. If the answer is No or If you don't know the answer, say 'NA/None'")
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
                                            that supports your answer. If the answer is No or If you don't know the answer, say 'NA/None'")
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
                                            from the document that supports your answer. If the answer is No or If you don't know the answer, say 'NA/None'")
                response[name]['novel drug formulations'] = {'Yes/No': output_llm.split('/')[0].strip(), 'sentence': output_llm.split('/')[1]}
                print("-----")

                # Process 'lead small-molecule drug candidates'
                output_llm = llm.query_rag("Does the document mention identifying to developing lead small-molecule drug candidates?\
                                        Example sentences for new lead small-molecule drug candidates:\
                                        1. Discovery of a Small Molecule Drug Candidate for Selective NKCC1 Inhibition in Brain Disorders\
                                        2. We have setup a drug discovery program of small-molecule compounds that act as chaperones enhancing\
                                            TTR/Amyloid-beta peptide (Aβ) interactions.\
                                            Answer either 'Yes' or 'No' followed by a '/' then the exact sentence without any changes from\
                                            the document that supports your answer. If the answer is No or If you don't know the answer, say 'NA/None'")
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
                                            from the document that supports your answer. If the answer is No or If you don't know the answer, say 'NA/None'")
                response[name]['clinical trials'] = {'Yes/No': output_llm.split('/')[0].strip(), 'sentence': output_llm.split('/')[1]}
                print("-----")
                
                #with open(f"files_output/{name}_llama3.1:70b_openai.json", "w") as f:
                    #json.dump(response, f, indent=4)
            except Exception as e:
                print(e)
                continue


    return str(response)



BOT_TOKEN = "xoxb-1089129130001-7130503874147-1eJXyg9HdahYaxOHANd8iyc0"
APP_TOKEN = "xapp-1-A074H63F6EL-7837148342518-b5be6962fadd66118e88e8aa2c1cdca5f393404870ef5d49ee25d944fc6fc02b"

app = App(token=BOT_TOKEN)
waiting_for_file = {}
questions = {}


@app.command("/help")
def help(ack, body, client):
    ack()
    user_id = body["user_id"]
    channel_id = body["channel_id"]
    client.chat_postMessage(
        channel=channel_id,
        text="I am a bot that can process PDF files. To get started, type `/processpdf`."
    )
    
    

    
@app.command("/get_authors_list")
def get_authors_list(ack, body, client):
    ack()
    user_id = body["user_id"]
    channel_id = body["channel_id"]
    
    with open("authors.txt", "r") as f: # To add file
        authors = f.read()
    
    client.chat_postMessage(
        channel=channel_id,
        text=f"Here's the authos list: {authors}"
    )
    

@app.command("/processpdf")
def handle_processpdf(ack, body, client):

    ack()
    user_id = body["user_id"]
    channel_id = body["channel_id"]
    questions[user_id] = body["text"].split("/")
    

    waiting_for_file[user_id] = channel_id
    client.chat_postMessage(
        channel=channel_id,
        text="Please upload the PDF file you'd like to process. If you have custom questions you want to ask the LLM, please write them next to the command seperated by a / otherwise leave it blank"
    )
    
@app.command("/hello")
def hello_command(ack, body):
    print("hello")
    user_id = body["user_id"]
    ack(f"Hi, <@{user_id}>!")

@app.event("app_mention")
def handle_app_mention(event, say):
    user = event["user"]
    user_text = event["text"]
    
    llm = Ollama(model="llama3.1")

    prompt = PromptTemplate(
        template="You are a text assistant, and here is someone asking you a question. Please provide a response. {question}",
        input_variables=["question"]
    )

    chain = prompt | llm
    response = chain.invoke({"question": user_text})
    say(f"Hi there, <@{user}>! \n {response}")

    # Add LLM
    


@app.event("file_shared")
def handle_file_shared(event, client, logger):
    user_id = event.get("user_id")
    file_id = event.get("file_id")
    
    if user_id in waiting_for_file:
        channel_id = waiting_for_file[user_id]
        try: user_questions = questions[user_id]
        except: user_questions = []
        del waiting_for_file[user_id]
        try:
            file_info_response = client.files_info(file=file_id)
            file_info = file_info_response["file"]
            if file_info["filetype"] == "pdf":
                url_private = file_info["url_private_download"]
                file_name = file_info["name"]
                response = requests.get(url_private, headers={'Authorization': f'Bearer {BOT_TOKEN}'})
                with open("papers/"+file_name, "wb") as f:
                    f.write(response.content)

                
                response = to_run()
                #spock = Spock(custom_questions=user_questions)
                #Spock()
                
                # Process the PDF file as needed
                # For example, extract text, analyze content, etc.
                # ...
                
                

                # Notify the user
                
                
                
                
                client.chat_postMessage(
                    channel=channel_id,
                    text=f"Your file `{file_name}` has been processed. \ {response}"
                )
            else:
                # Not a PDF file
                client.chat_postMessage(
                    channel=channel_id,
                    text="The uploaded file is not a PDF. Please try again."
                )
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            client.chat_postMessage(
                channel=channel_id,
                text=f"An error occurred while processing your file. Please try again. {e}"
            )
    else:
        # User is not expecting to upload a file
        pass  # Optionally handle unexpected file uploads

if __name__ == "__main__":
    # Initialize Socket Mode handler
    handler = SocketModeHandler(app, APP_TOKEN)
    handler.start()
