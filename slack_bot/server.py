from langchain_community.vectorstores import FAISS
import faiss


import os
import requests
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
#from  ../spock_
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

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

                
                #response = to_run()
                spock = Spock(paper="papers/"+file_name,custom_questions=user_questions)
                response = Spock()
                
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
