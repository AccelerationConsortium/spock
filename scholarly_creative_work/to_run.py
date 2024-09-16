from LLM import LLM
import os
import json
from utilities import Bot_LLM


pdf_list = os.listdir("papers/papers")
print(pdf_list)
response = {}
format_instruction = "Output  either 'Yes' or 'No' followed by a '/' then a sentence from the document that supports your answer."
for i in range(10):
    response[pdf_list[i].split('/')[-1].replace("_","/")] = {}
    llm = Bot_LLM(folder_path='db/db'+str(i))
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
    output_llm = llm.query_rag("Does the article mention any new material discovery? Output either 'Yes' or 'No' followed by a '/' then a sentence from the document that supports your answer.")
    response[i]['new materials'] = {'Yes/No': output_llm.split('/')[0].strip(), 'sentence': output_llm.split('/')[1]}
    print("-----")

    # Process 'screening algorithms'
    output_llm = llm.query_rag("A screening algorithm is a systematic procedure or method used to identify individuals who may have or be at risk for a particular condition or trait within a large population. These algorithms are designed to quickly and efficiently screen out those who are unlikely to have the condition, while identifying those who may require further diagnostic evaluation or intervention. If there are any, what are the screening algorithms used in the paper? Output either 'Yes' or 'No' followed by a '/' then a sentence from the document that supports your answer.")
    response[i]['screening algorithms'] = {'Yes/No': output_llm.split('/')[0].strip(), 'sentence': output_llm.split('/')[1]}
    print("-----")

    # Process 'AI algorithms'
    output_llm = llm.query_rag("AI algorithms are computational methods and processes used to solve specific tasks by mimicking human intelligence. These algorithms enable machines to learn from data, make decisions, and perform tasks that typically require human intelligence. Output either 'Yes' or 'No' followed by a '/' then a sentence from the document that supports your answer.")
    response[i]['AI algorithms'] = {'Yes/No': output_llm.split('/')[0].strip(), 'sentence': output_llm.split('/')[1]}
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
    response[i]['models'] = {'Yes/No': output_llm.split('/')[0].strip(), 'sentence': output_llm.split('/')[1]}
    print("-----")

    # Process 'funding'
    output_llm = llm.query_rag("Does the article mention who funded it? Output either 'Yes' or 'No' followed by a '/' then a sentence from the document that supports your answer.")
    response[i]['funding'] = {'Yes/No': output_llm.split('/')[0].strip(), 'sentence': output_llm.split('/')[1]}
    print("-----")

    # Process 'material datasets'
    output_llm = llm.query_rag("Does the article share any AI or material-related datasets? If yes, provide the details. Output either 'Yes' or 'No' followed by a '/' then a sentence from the document that supports your answer.")
    response[i]['material datasets'] = {'Yes/No': output_llm.split('/')[0].strip(), 'sentence': output_llm.split('/')[1]}
    print("-----")

    # Process 'drug formulations explored'
    output_llm = llm.query_rag("Has the article explored any new drug formulations? If yes, what are they? Output either 'Yes' or 'No' followed by a '/' then a sentence from the document that supports your answer.")
    response[i]['drug formulations explored'] = {'Yes/No': output_llm.split('/')[0].strip(), 'sentence': output_llm.split('/')[1]}
    print("-----")

    # Process 'novel drug formulations'
    output_llm = llm.query_rag("Does the article identify any novel drug formulations? If yes, provide the details. Output either 'Yes' or 'No' followed by a '/' then a sentence from the document that supports your answer.")
    response[i]['novel drug formulations'] = {'Yes/No': output_llm.split('/')[0].strip(), 'sentence': output_llm.split('/')[1]}
    print("-----")

    # Process 'lead small-molecule drug candidates'
    output_llm = llm.query_rag("Does the article mention any lead small-molecule drug candidates? If so, what are they? Output either 'Yes' or 'No' followed by a '/' then a sentence from the document that supports your answer.")
    response[i]['lead small-molecule drug candidates'] = {'Yes/No': output_llm.split('/')[0].strip(), 'sentence': output_llm.split('/')[1]}
    print("-----")

    # Process 'clinical trials'
    output_llm = llm.query_rag("Are there any clinical trials mentioned in the article? If yes, provide the details. Output either 'Yes' or 'No' followed by a '/' then a sentence from the document that supports your answer.")
    response[i]['clinical trials'] = {'Yes/No': output_llm.split('/')[0].strip(), 'sentence': output_llm.split('/')[1]}
    print("-----")

    print(response)

with open("llm_ouput.json", "w") as f:
    json.dump(response, f)
