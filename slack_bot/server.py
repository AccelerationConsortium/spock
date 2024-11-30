from langchain_community.vectorstores import FAISS
import faiss
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
import requests
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from spock_literature import Spock
from spock_literature.utils.Author import Author
import json
from User import User
from spock_literature.utils.generate_podcast import generate_audio
import logging
import subprocess
from langchain_community.callbacks import get_openai_callback


load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
APP_TOKEN = os.getenv("APP_TOKEN")
PAPERS_PATH = "/home/m/mehrad/brikiyou/scratch/spock/slack_bot/papers/"
USER_JSON_PATH = "/home/m/mehrad/brikiyou/scratch/spock/slack_bot/users.json"
ANALYZED_PAPERS_JSON_PATH = "/home/m/mehrad/brikiyou/scratch/spock/slack_bot/analyzed_publications.json"
SCRIPTS_PATH = "/home/m/mehrad/brikiyou/scratch/spock/slack_bot/scripts/"


app = App(token=BOT_TOKEN)
waiting_for_file = {}
waiting_for_podcast = {}
questions = {}


@app.command("/help")
def help(ack, body, client):
    ack()
    user_id = body["user_id"]
    channel_id = body["channel_id"]
    text = body["text"] # Add LLM here to help
    client.chat_postMessage(
        channel=channel_id,
        text="""👋 Welcome! I'm Spock 🖖, a bot designed to process PDF files and assist you in extracting valuable information from publications. To get started, simply type `/processpdf`.

Commands:

- 📜 get_authors_list: Retrieve a list of all authors.
  Example: /get_authors_list

- 📚 get_author_publication: Get the latest publications of a specific author. By default, it fetches the most recent one. To specify how many publications you'd like, provide the author's name followed by a comma and the number of publications.
  Example: /get_author_publication Jane Doe, 3 (This will retrieve the 3 latest publications by Jane Doe.)

- 📄 process_pdf: Share and process a PDF file from your local computer. You can also include custom questions about the PDF by writing them next to the command, separated by a `/`.
  Example: /process_pdf Does the paper discuss the impact of AI on society? / What are the key findings of the paper?

- 🔗 process_publication_doi: Process a publication using its DOI (Digital Object Identifier). Just enter the DOI after the command.
  Example: /process_publication_doi example-doi

- 📝 process_publication_title: Process a publication by its title. Simply enter the title after the command.
  Example: /process_publication_title The Future of AI in 2024
  
- 🤖 choose_llm: Choose the language model you'd like to use for future responses. You can choose between Llama3.1, Claude 3.5 Sonnet, and GPT-4. To do so, type /choose_llm followed by the model name.

- 🎙️ generate_podcast: Generate a podcast from a text input. This command is currently disabled.
Feel free to ask me questions or share your files for processing!

        """
    )
    
#TODO
# - deleting pdfs after some time if unuesed - to implement
# - Storing all the pdfs chunks in one single vectorestore and haveing the user query that vectorstore through the @app_mention
# - Choose the LLM with a /command  | LLama | Claude 3.5 Sonnet | OpenAI gpt 4-o (High priority - fixed (need claud key))
# - https://www.nvidia.com/en-us/ai/
# - OpenAI CallBack for trqcking tokens - To implement
# - Add github actions and testing 
# - Storage - To upgrade
# - Upgrading quality of the summaries - fixed



# Slack event handler
@app.command("/generate_podcast")
def handle_app_mention(ack, body, client):
    ack()
    channel_id = body['channel_id']
    user_id = body['user_id']
    waiting_for_podcast[user_id] = channel_id
    client.chat_postMessage(
        channel=channel_id,
        text="Please upload the PDF file you'd like to convert to a podcast."
    )
    



@app.command("/process_publication_title")
def handle_process_publication_name(ack, body, client):
    ack()
    user = body["user_id"]
    channel_id = body["channel_id"]
    title = body["text"]
    
    print(title)
    
    
    spock = Spock(publication_title=title, path=PAPERS_PATH)
    try: 
        response = f" Hi there, <@{user}>! Here is the summary of the publication with the title {title}: {spock()}"
    except Exception as e:
        if isinstance(e, RuntimeError):
            response = "Couldn't download the PDF, probably because the publication is not available online."
        else:
            response = "An error occurred while processing the publication."
            
    
    client.chat_postMessage(
        channel=channel_id,
        text=f"{response}"
    )
    

    
    
@app.command("/process_publication_doi")
def handle_process_publication_doi(ack, body, client):
    ack()
    user = body["user_id"]
    channel_id = body["channel_id"]
    doi = body["text"]
    
    print(doi)
    
    
    spock = Spock(publication_doi=doi, path=PAPERS_PATH)
    try: 
        response = f" Hi there, <@{user}>! Here is the summary of the publication with the DOI {doi}: {spock()}"
    except Exception as e:
        if isinstance(e, RuntimeError):
            response = "Couldn't download the PDF, probably because the publication is not available online."
        else:
            response = "An error occurred while processing the publication."
            
    
    client.chat_postMessage(
        channel=channel_id,
        text=f"{response}"
    )
    

# TODO: Better formatting
@app.command("/get_authors_list")
def get_authors_list(ack, body, client):
    ack()
    user_id = body["user_id"]
    channel_id = body["channel_id"]
    
    with open("authors.txt", "r") as f: 
        authors = f.read()
    
    client.chat_postMessage(
        channel=channel_id,
        text=f"Here's the authors list: {authors}"
    )
    
@app.command("/get_author_publication")
def get_author_publications(ack, body, client):
    ack()
    user_id = body["user_id"]
    channel_id = body["channel_id"]
    text = body["text"]
    try: 
        author, count = text.split(",") 
    except:
        author = text
        count = 1
        
    author = Author(author)
    # Format publications
    publications = author(int(count))
    
    output = ""
    for author, works in publications.items():
        for work in works:
            output += f"📄 Title: {work['title']}\n"
            output += "📜 Abstract:\n"
            output += f"    {work['abstract']}\n"
            output += f"✍️ Authors: {work['author']}\n"
            output += f"📅 Year: {work['year']}\n"
            output += f"🔗 URL: {work['url']}\n"
            output += "\n" + "-"*50 + "\n\n"

    
    client.chat_postMessage(
        channel=channel_id,
        text=f"Here are the {count} latest publications of {author}: {output}"
    )
    
        
        
        
@app.command("/get_user_settings")
def get_user_settings(ack, body, client):
    ack()
    user_id = body["user_id"]
    channel_id = body["channel_id"]
    
    
    

@app.command("/choose_llm")
def handle_choose_llm(ack, body, client):
    ack()
    user_id = body["user_id"]
    channel_id = body["channel_id"]
    model = body["text"].strip().lower()
    print(model)
    
    with open(USER_JSON_PATH, "r") as f:
        users = json.load(f)
    
    if user_id not in users:
        user = User(user_id, model)
        users[user_id] = user.__dict__()[user.user_id]
    else:
        users[user_id]["user_model"] = model
        
    with open(USER_JSON_PATH, "w") as f:
        json.dump(users, f)
    
    
    client.chat_postMessage(
        channel=channel_id,
        text=f"Hi there, <@{user_id}>! You've chosen the {model} model. I'll use this model for future responses."
    )



@app.command("/get_history") # To edit
def handle_get_history(ack, body, client):
    ack()
    user_id = body["user_id"]
    channel_id = body["channel_id"]
    
    with open(ANALYZED_PAPERS_JSON_PATH, "r") as f:
        analyzed_papers = json.load(f)
    
    user_files = list(filter(lambda x: x["user_id"]==user_id, analyzed_papers))
    
    if not user_files:
        client.chat_postMessage(
            channel=channel_id,
            text="You haven't processed any files yet."
        )
        return
    
    output = ""
    for file in user_files:
        output += f"\n📄 Title: {file['title']}"    
    client.chat_postMessage(
        channel=channel_id,
        text=f"Here are the files you've processed: {output}"
    )
@app.command("/process_pdf")
def handle_processpdf(ack, body, client):

    ack()
    user_id = body["user_id"]
    channel_id = body["channel_id"]
    questions[user_id] = list(filter(lambda x:x, body["text"].split("/")))
    

    waiting_for_file[user_id] = channel_id
    client.chat_postMessage(
        channel=channel_id,
        text="Please upload the PDF file you'd like to process.",
        mrkdwn=True

    )
    
@app.event("app_mention")
def handle_app_mention(event, say):
    user = event["user"]
    user_text = event["text"]
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.4)


    # Customize it 
    prompt = PromptTemplate(
        template="You are a text assistant, and here is someone asking you a question. Please provide a response. {question}",
        input_variables=["question"]
    )

    chain = prompt | llm
    response = chain.invoke({"question": user_text})
    say(f"Hi there, <@{user}>! \n {response}")

    


@app.event("file_shared")
def handle_file_shared(event, client, logger):
    user_id = event.get("user_id")
    file_id = event.get("file_id")
    
    if user_id in waiting_for_podcast:
        channel_id = waiting_for_podcast[user_id]
        try: user_questions = questions[user_id]
        except: user_questions = []
        del waiting_for_podcast[user_id]
        try:
            file_info_response = client.files_info(file=file_id)
            file_info = file_info_response["file"]
            if file_info["filetype"] == "pdf":
                url_private = file_info["url_private_download"]
                file_name = file_info["name"]
                response = requests.get(url_private, headers={'Authorization': f'Bearer {BOT_TOKEN}'})
                print(f"Writing {file_name} to disk")
                with open(PAPERS_PATH+file_name, "wb") as f:
                    f.write(response.content)
                    
                client.chat_postMessage(
                    channel=channel_id,
                    text=f"Your file `{file_name}` has been uploaded. Processing it now..."
                )
                script_path = SCRIPTS_PATH+"submit_generate_podcast.sh"
                args = [script_path, PAPERS_PATH + file_name, user_id, channel_id, "Here's the audio podcast for your pdf!"]

                try:
                    subprocess.run(args, check=True)
                    print("Script executed successfully!")
                except subprocess.CalledProcessError as e:
                    print(f"An error occurred while executing the script: {e}")
                    print(f"stderr: {e.stderr}")
                    print(f"stdout: {e.stdout}")

                

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
        


    
    elif user_id in waiting_for_file:
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
                with open(PAPERS_PATH+file_name, "wb") as f:
                    f.write(response.content)
                    
                client.chat_postMessage(
                    channel=channel_id,
                    text=f"Your file `{file_name}` has been uploaded. Processing it now..."
                )

                print(f"passing {file_name} to Spock")
                
                
                
                with open(USER_JSON_PATH, "r") as f:
                    users = json.load(f)
                    
                
                if user_id not in users:
                    user = User(user_id, "llama3.1")
                    users[user_id] = user.__dict__()[user.user_id]
                    with open(USER_JSON_PATH, "w") as f:
                        json.dump(users, f)
                
                
                # Choosing the model for the user
                model = users[user_id]["user_model"]
                
                with get_openai_callback() as cb:
                    
                    spock = Spock(model=model,paper=PAPERS_PATH+file_name,custom_questions=user_questions)                
                    spock()
                    response_output = spock.format_output()
                    cost = cb.total_cost
                    """
                    args = []
                    script_path = SCRIPTS_PATH+"submit_process_pdf.sh"
                    try:
                        subprocess.run(args, check=True)
                        print("Script executed successfully!")
                    except subprocess.CalledProcessError as e:
                        print(f"An error occurred while executing the script: {e}")
                        print(f"stderr: {e.stderr}")
                        print(f"stdout: {e.stdout}")
                    """
                
                
                
                # Sending the response to the user
                client.chat_postMessage(
                    channel=channel_id,
                    text=f"Your file `{file_name}` has been processed. \n {response_output} \n Cost (USD): {cost}",
                    mrkdwn=True
                )
                """
                
                """
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
        pass  

if __name__ == "__main__":
    # Initialize Socket Mode handler
    handler = SocketModeHandler(app, APP_TOKEN)
    handler.start()
