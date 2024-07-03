import json
import time
import concurrent.futures
from author import Author
from bot import Bot_LLM


if __name__ == "__main__":
    bot_llm = Bot_LLM()
    
    
    with open('json/output.json','r') as file:
        data = json.load(file)
        
    for author in data:
        try:
            author_filled = Author(author)
            print('Author created successfully for ' + author)
            author_filled.setup_author('json/output.json', llm=bot_llm)
            print(f"Topics for {author} have been updated")
        except Exception as e:
            print(f"Couldn't find the google scholar profile for {author}: {e}")
        
    ### RAG
    
    with open('json/output.json','r') as file:
        data = json.load(file)
    for author in data:
        if data[author]['pdf'] != None:
            try:
                bot_llm.rag(data[author]['pdf']) 
                print(f"RAG for {author} pdf generated successfully")
            except Exception as e:
                print(f"Couldn't generate the RAG for {author} PDF : {e}")
        else:
            try:
                bot_llm.rag(data[author]['abstract']) 
                print(f"RAG for {author} abstract generated successfully")
            except Exception as e:
                print(f"Couldn't generate the RAG for {author} abstract: {e}")
                
                
    ### Query 
                      