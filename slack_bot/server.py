from langchain_community.vectorstores import FAISS
import faiss
from dotenv import load_dotenv
import os
import requests
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from spock_literature import Spock
from spock_literature.classes.Author import Author

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
APP_TOKEN = os.getenv("APP_TOKEN")

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
        text="""I am a bot that can process PDF files. To get started, type `/processpdf`.   \ 
        
        
        Commands:
        
        - get_authors_list: Get the list of authors.
        - get_author_publication: Get the x latest publications of an author. Default is 1. To choose how many publications you want to get, write the name of the author seperated by a comma next to the values of the x latest artcicles you want.
        - process_pdf: Process a PDF file. To process a PDF file, share the file you have locally on your computer, possibly write your custom questions next to the command seperated by a /.
        - process_publication_doi: Process a publication by its DOI. To process a publication by its DOI, write the DOI of the publication next to the command.
        - process_publication_title: Process a publication by its name. To process a publication by its name, write the name of the publication next to the command.
        """
    )
    
#TODO
# - deleting pdfs after some time if unuesed



@app.command("/process_publication_title")
def handle_process_publication_name(ack, body, client):
    ack()
    user_id = body["user_id"]
    channel_id = body["channel_id"]
    name = body["text"]
    
    """
    spock = Spock(name=name)
    response = spock()
    client.chat_postMessage(
        channel=channel_id,
        text=f"Here is the summary of the publication with name {name}: {response}"
    )
    """
@app.command("/process_publication_doi")
def handle_process_publication_doi(ack, body, client):
    ack()
    user_id = body["user_id"]
    channel_id = body["channel_id"]
    doi = body["text"]
    
    
    """
    spock = Spock(doi=doi)
    response = spock()
    client.chat_postMessage(
        channel=channel_id,
        text=f"Here is the summary of the publication with DOI {doi}: {response}"
    )
    
    """
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
    
@app.command("/get_author_publication")
def get_author_publications(ack, body, client):
    ack()
    user_id = body["user_id"]
    channel_id = body["channel_id"]
    text = body["text"]
    author, count = text.split(",") # To edit 
    author = Author(author)
    # Format publications
    publications = author(int(count))
    publications = str(publications) # To format this dict
    
    
    client.chat_postMessage(
        channel=channel_id,
        text=f"Here are the {count} latest publications of {author}: {publications}"
    )
    
        
        


@app.command("/process_pdf")
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
                print(f"Writing {file_name} to disk")
                with open("/home/m/mehrad/brikiyou/scratch/spock_package/spock/slack_bot/papers/"+file_name, "wb") as f:
                    f.write(response.content)

                print(f"passing {file_name} to Spock")
                
                # Analyzing the paper
                spock = Spock(paper="/home/m/mehrad/brikiyou/scratch/spock_package/spock/slack_bot/papers/"+file_name,custom_questions=user_questions)
                response = spock()
                
                # Sending the response to the user
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
        pass  # Optionally handle unexpected file uploads

if __name__ == "__main__":
    # Initialize Socket Mode handler
    handler = SocketModeHandler(app, APP_TOKEN)
    handler.start()
