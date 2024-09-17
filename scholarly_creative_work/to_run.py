from LLM import LLM
import os
import json
from utilities import Bot_LLM

with open("llm_output.json", "r") as f:
    response = json.load(f)
    
    
    
pdf_list = os.listdir("papers/papers")
#print(pdf_list)
#response = {}
format_instruction = "Output  either 'Yes' or 'No' followed by a '/' then a sentence from the document that supports your answer. If you don't know the answer, right 'NA/'"
for i in range(5):
    #name = pdf_list[i].split('.')[-1].split('/')[-1].replace("_","/")
    
    name = pdf_list[i]
    print(name)
    if not os.path.exists(f'files_output/{name}'):
        try:
            response[name] = {}
            llm = Bot_LLM(model="llama3.1", folder_path='db/db'+str(i))
            llm.chunk_indexing("papers/papers/"+pdf_list[i])    


            
            #llm.set_folder_path("db/db"+str(i))
            
            
            print("running split_and_embedding_chunk_pdf")
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

            # Process 'new materials'
            
            output_llm = llm.query_rag("Does the article mention any new material discovery? Output either 'Yes' or 'No' followed by a '/' then an exact sentence without any changes from the document that supports your answer.")
            response[name]['new materials'] = {'Yes/No': output_llm.split('/')[0].strip(), 'sentence': output_llm.split('/')[1]}
            print("-----")

            # Process 'screening algorithms'
            output_llm = llm.query_rag("A screening algorithm is a systematic procedure or method used to identify individuals who may have or be at risk for a particular condition or trait within a large population. These algorithms are designed to quickly and efficiently screen out those who are unlikely to have the condition, while identifying those who may require further diagnostic evaluation or intervention. If there are any, what are the screening algorithms used in the paper? Output either 'Yes' or 'No' followed by a '/' then a sentence from the document that supports your answer.")
            response[name]['screening algorithms'] = {'Yes/No': output_llm.split('/')[0].strip(), 'sentence': output_llm.split('/')[1]}
            print("-----")

            # Process 'AI algorithms' To edit
            output_llm = llm.query_rag("AI algorithms are computational methods and processes used to solve specific tasks by mimicking human intelligence. These algorithms enable machines to learn from data, make decisions, and perform tasks that typically require human intelligence. Output either 'Yes' or 'No' followed by a '/' then a sentence from the document that supports your answer.")
            response[name]['AI algorithms'] = {'Yes/No': output_llm.split('/')[0].strip(), 'sentence': output_llm.split('/')[1]}
            print("-----")

            # Process 'workflow'
            '''
            output_llm = llm.query_rag("Can you describe to me the workflow used by the author? Output either 'Yes' or 'No' followed by a '/' then a sentence from the document that supports your answer.")
            response[i]['workflow'] = {'Yes/No': output_llm.split('/')[0].strip(), 'sentence': output_llm.split('/')[1]}
            print("-----")

            # Process 'methods'
            output_llm = llm.query_rag("Can you do a methods description? Output either 'Yes' or 'No' followed by a '/' then a sentence from the document that supports your answer.")
            response[i]['methods'] = {'Yes/No': output_llm.split('/')[0].strip(), 'sentence': output_llm.split('/')[1]}
            print("-----")
            '''
            # Process 'models'
            output_llm = llm.query_rag("What are the models used in the article? Output either 'Yes' or 'No' followed by a '/' then a sentence from the document that supports your answer.")
            response[name]['models'] = {'Yes/No': output_llm.split('/')[0].strip(), 'sentence': output_llm.split('/')[1]}
            print("-----")

            # Process 'funding' 
            # Add more context to the question/ check acknowlegdment section
            output_llm = llm.query_rag("Does the article mention who funded it? Output either 'Yes' or 'No' followed by a '/' then a sentence from the document that supports your answer.")
            response[name]['funding'] = {'Yes/No': output_llm.split('/')[0].strip(), 'sentence': output_llm.split('/')[1]}
            print("-----")

            # Process 'material datasets'
            output_llm = llm.query_rag("Does the article share any AI or material-related datasets? If yes, provide the details. Output either 'Yes' or 'No' followed by a '/' then a sentence from the document that supports your answer.")
            response[name]['material datasets'] = {'Yes/No': output_llm.split('/')[0].strip(), 'sentence': output_llm.split('/')[1]}
            print("-----")

            # Process 'drug formulations explored'
            output_llm = llm.query_rag("Has the article explored any new drug formulations? If yes, what are they? Output either 'Yes' or 'No' followed by a '/' then a sentence from the document that supports your answer.")
            response[name]['drug formulations explored'] = {'Yes/No': output_llm.split('/')[0].strip(), 'sentence': output_llm.split('/')[1]}
            print("-----")

            # Process 'novel drug formulations'
            output_llm = llm.query_rag("Does the article identify any novel drug formulations? If yes, provide the details. Output either 'Yes' or 'No' followed by a '/' then a sentence from the document that supports your answer.")
            response[name]['novel drug formulations'] = {'Yes/No': output_llm.split('/')[0].strip(), 'sentence': output_llm.split('/')[1]}
            print("-----")

            # Process 'lead small-molecule drug candidates'
            output_llm = llm.query_rag("Does the article mention any lead small-molecule drug candidates? If so, what are they? Output either 'Yes' or 'No' followed by a '/' then a sentence from the document that supports your answer.")
            response[name]['lead small-molecule drug candidates'] = {'Yes/No': output_llm.split('/')[0].strip(), 'sentence': output_llm.split('/')[1]}
            print("-----")

            # Process 'clinical trials'
            output_llm = llm.query_rag("Are there any clinical trials mentioned in the article? If yes, provide the details. Output either 'Yes' or 'No' followed by a '/' then a sentence from the document that supports your answer.")
            response[name]['clinical trials'] = {'Yes/No': output_llm.split('/')[0].strip(), 'sentence': output_llm.split('/')[1]}
            print("-----")
            os.mknod(f"{name}.json") 
            with open(f"files_output/{name}.json", "w") as f:
                json.dump(response, f, indent=4)
        except Exception as e:
            print(e)
            continue


print(response)

