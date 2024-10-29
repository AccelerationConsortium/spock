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
import json




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

Feel free to ask me questions or share your files for processing!

        """
    )
    
#TODO
# - deleting pdfs after some time if unuesed
# - Maybe addind PDF files that were done and their responses in a database
# - Storing all the pdfs chunks in one single vectorestore and haveing the user query that vectorstore through the @app_mention



@app.command("/process_publication_title")
def handle_process_publication_name(ack, body, client):
    ack()
    user = body["user_id"]
    channel_id = body["channel_id"]
    title = body["text"]
    
    print(title)
    
    
    spock = Spock(publication_title=title)
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
    
    
    spock = Spock(publication_doi=doi)
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
    
    
@app.command("/get_authors_list")
def get_authors_list(ack, body, client):
    ack()
    user_id = body["user_id"]
    channel_id = body["channel_id"]
    
    with open("authors.txt", "r") as f: 
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
    
        
        


@app.command("/process_pdf")
def handle_processpdf(ack, body, client):

    ack()
    user_id = body["user_id"]
    channel_id = body["channel_id"]
    questions[user_id] = list(filter(lambda x:x, body["text"].split("/")))
    
    print(questions[user_id])
    print(questions)

    waiting_for_file[user_id] = channel_id
    client.chat_postMessage(
        channel=channel_id,
        text="Please upload the PDF file you'd like to process."
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
                with open("/home/m/mehrad/brikiyou/scratch/spock_package/spock/slack_bot/analyzed_publications.json", "r") as f:
                    analyzed_papers = json.load(f)
                    
                if file_name in analyzed_papers:
                    client.chat_postMessage(
                        channel=channel_id,
                        text=f"Hi there, <@{user_id}>! This paper has already been processed. Here is the summary: {analyzed_papers[file_name]}"
                    )
                    return
                
                
                spock = Spock(paper="/home/m/mehrad/brikiyou/scratch/spock_package/spock/slack_bot/papers/"+file_name,custom_questions=user_questions)
                response = spock()
                
                
                with open("/home/m/mehrad/brikiyou/scratch/spock_package/spock/slack_bot/analyzed_publications.json", "w") as f: # TODO: Change this to match custom questions
                    analyzed_papers[file_name] = response
                    json.dump(analyzed_papers, f)
                
                
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
        pass  

if __name__ == "__main__":
    # Initialize Socket Mode handler
    handler = SocketModeHandler(app, APP_TOKEN)
    handler.start()
