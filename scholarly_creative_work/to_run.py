from LLM import LLM
import os
import json


pdf_list = os.listdir("papers/papers")
print(pdf_list)
response = {}
format_instruction = "Output  either 'Yes' or 'No' followed by a '/' then a sentence from the document that supports your answer."
for i in range(10):
    response[i] = {}
    llm = LLM()
    llm.set_folder_path("db/db"+str(i))
    
    
    print("running split_and_embedding_chunk_pdf")
    llm.split_and_embedding_chunk_pdf("papers/papers/"+pdf_list[i])
    response[i]['affiliation'] = llm.query_rag("What are the authors affiliation. Output a dictionary ? Output  either 'Yes' or 'No' followed by a '/' then a sentence from the document that supports your answer.")
    print("-----")
    response[i]['topic']  = llm.query_rag("What are topics of the paper ?Output  either 'Yes' or 'No' followed by a '/' then a sentence from the document that supports your answer.")
    print("-----")
    response[i]['new materials'] = llm.query_rag("does the article mention any new material discovery ? Output  either 'Yes' or 'No' followed by a '/' then a sentence from the document that supports your answer.")
    print("-----")
    response[i]['screening algorithms'] = llm.query_rag("A screening algorithm is a systematic procedure or method used to identify individuals who may have or be at risk for a particular condition or trait within a large population. These algorithms are designed to quickly and efficiently screen out those who are unlikely to have the condition, while identifying those who may require further diagnostic evaluation or intervention. If there are any, What are the screening algorithms used in the paper ? Output  either 'Yes' or 'No' followed by a '/' then a sentence from the document that supports your answer.")
    print("-----")
    response[i]['AI algorithms'] = llm.query_rag("AI algorithms are computational methods and processes used to solve specific tasks by mimicking human intelligence. These algorithms enable machines to learn from data, make decisions, and perform tasks that typically require human intelligence. Output  either 'Yes' or 'No' followed by a '/' then a sentence from the document that supports your answer.")
    print("-----")
    
    response[i]['workflow'] = llm.query_rag("Can you describe to me the workflow used by the author ? Output  either 'Yes' or 'No' followed by a '/' then a sentence from the document that supports your answer.")
    print("-----")
    response[i]['methods'] = llm.query_rag("can you do a methods description ? Output  either 'Yes' or 'No' followed by a '/' then a sentence from the document that supports your answer.")
    print("-----")
    response[i]['models'] = llm.query_rag("what are the models used in the article ? Output  either 'Yes' or 'No' followed by a '/' then a sentence from the document that supports your answer.")
    print("-----")
    response[i]['funding'] = llm.query_rag("does the article mention who funded it. Output  either 'Yes' or 'No' followed by a '/' then a sentence from the document that supports your answer.")

    #response[i]['new composition of matter'] = llm.query_rag("Has the article discovered any new composition of matter? If yes, please provide details.")
    #print("-----")
    #response[i]['AI or screening algorithms'] = llm.query_rag("Does the article describe any new AI or screening algorithms, methods, workflows, or models? If yes, provide details.")
    #print("-----")
    #response[i]['experimental methodologies'] = llm.query_rag("Are there any new experimental methodologies published in the article? Please provide a description.")
    #print("-----")
    response[i]['material datasets'] = llm.query_rag("Does the article share any AI or material-related datasets? If yes, provide the details. Output  either 'Yes' or 'No' followed by a '/' then a sentence from the document that supports your answer.")
    print("-----")
    response[i]['drug formulations explored'] = llm.query_rag("Has the article explored any new drug formulations? If yes, what are they? Output  either 'Yes' or 'No' followed by a '/' then a sentence from the document that supports your answer.")
    print("-----")
    response[i]['novel drug formulations'] = llm.query_rag("Does the article identify any novel drug formulations? If yes, provide the details. Output  either 'Yes' or 'No' followed by a '/' then a sentence from the document that supports your answer.")
    print("-----")
    response[i]['lead small-molecule drug candidates'] = llm.query_rag("Does the article mention any lead small-molecule drug candidates? If so, what are they? Output  either 'Yes' or 'No' followed by a '/' then a sentence from the document that supports your answer.")
    print("-----")
    response[i]['clinical trials'] = llm.query_rag("Are there any clinical trials mentioned in the article? If yes, provide the details. Output  either 'Yes' or 'No' followed by a '/' then a sentence from the document that supports your answer.")
    print("-----")
    print(response)

with open("llm_ouput.json", "w") as f:
    json.dump(response, f)
