from dotenv import load_dotenv
import os
import requests
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import json
from User import User
import subprocess
from texts import *
from pathlib import Path




load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
APP_TOKEN = os.getenv("APP_TOKEN")

def create_empty_sh_file(file_path):
    """
    Create an empty shell script file
    """
    path = Path(file_path) if isinstance(file_path, str) else file_path
    path.touch(exist_ok=False)
    path.chmod(0o755)
    return path

def count_files(dir=GENERATED_SCRIPTS_PATH):
    """
    Count the number of files in a directory
    """ 
    return len([1 for x in list(os.scandir(dir)) if x.is_file()]) 



app = App(token=BOT_TOKEN)
waiting_for_file = {}
waiting_for_podcast = {}
questions = {}


@app.command("/help")
def help(ack, body, client):
    ack()
    user_id = body["user_id"]
    channel_id = body["channel_id"]
    text = body["text"]
    client.chat_postMessage(
        channel=channel_id,
        text=f"""👋 Welcome! I'm Spock 🖖, a bot designed to process PDF files and assist you in extracting valuable information from publications. To get started, simply type `/processpdf`.

Commands: {COMMANDS} """
    )
    


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
    



@app.command("/process_publication")
def handle_process_publication_name(ack, body, client):
    ack()
    user = body["user_id"]
    channel_id = body["channel_id"]
    try: 
        response = body["text"].split("||")
        publication, custom_questions = response[0], "".join(response[1:])
    except: 
        publication = body["text"]
        custom_questions = ""
    script_path = SCRIPTS_PATH+"submit_process_publication.sh"
    generate_script = create_empty_sh_file(GENERATED_SCRIPTS_PATH+f"process_publication_{user}{count_files()}.sh")
    args = [script_path, publication, custom_questions, user, channel_id, generate_script]    
    try:
        subprocess.run(args, check=True)
        print("Script executed successfully!")
        client.chat_postMessage(
        channel=channel_id,
        text="Script Submitted Successfully! Processing your file now..."
    )

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing the script: {e}")

        
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

    try:
        script_path = SCRIPTS_PATH+"submit_get_author_publication.sh"
        generated_script = create_empty_sh_file(GENERATED_SCRIPTS_PATH+f"get_author_publication_{user_id}{count_files()}.sh")
        args = [script_path, author, str(count), user_id, channel_id, generated_script]
        subprocess.run(args, check=True)
        print("Script executed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing the script: {e}")
    
        
        
        
@app.command("/get_user_settings")
def get_user_settings(ack, body, client):
    ack()
    user_id = body["user_id"]
    channel_id = body["channel_id"]
    with open(USER_JSON_PATH, "r") as f:
        users = json.load(f)
    

    if user_id not in users:
        user = User(user_id, "llama3.3")
        users[user_id] = user.__dict__()[user.user_id]
        with open(USER_JSON_PATH, "w") as f:
            json.dump(users, f)
        
    client.chat_postMessage(
        channel=channel_id,
        text=f"Hi there, <@{user_id}>! Your current model is: {users[user_id]['user_model']}"
    )

    
@app.command("/toggle_binary_responses")
def turn_on_binary_responses(ack, body, client):
    ack()
    user_id = body["user_id"]
    channel_id = body["channel_id"]
    with open(USER_JSON_PATH, "r") as f:
        users = json.load(f)
    
    if user_id not in users:
        user = User(user_id, "llama3.3")
        users[user_id] = user.__dict__()[user.user_id]
    users[user_id]["settings"]["Binary Response"] = not users[user_id]["settings"]["Binary Response"]
     
    with open(USER_JSON_PATH, "w") as f:
        json.dump(users, f)
    
    client.chat_postMessage(
        channel=channel_id,
        text=f'Hi there, <@{user_id}>! Binary responses have been turned {"on" if users[user_id]["settings"]["Binary Response"] else "off"}.'
    )



@app.command("/toggle_summary")
def turn_on_summary(ack, body, client):
    ack()
    user_id = body["user_id"]
    channel_id = body["channel_id"]
    with open(USER_JSON_PATH, "r") as f:
        users = json.load(f)
    
    if user_id not in users:
        user = User(user_id, "llama3.3")
        users[user_id] = user.__dict__()[user.user_id]
    users[user_id]["settings"]["Summary"] = not users[user_id]["settings"]["Summary"]
     
    with open(USER_JSON_PATH, "w") as f:
        json.dump(users, f)
    
    client.chat_postMessage(
        channel=channel_id,
        text=f'Hi there, <@{user_id}>! Summaries have been turned {"on" if users[user_id]["settings"]["Summary"] else "off"}.'
    )


@app.command("/choose_llm")
def handle_choose_llm(ack, body, client):
    ack()
    user_id = body["user_id"]
    channel_id = body["channel_id"]
    model = body["text"].strip().lower()
    
    with open(USER_JSON_PATH, "r") as f:
        users = json.load(f)
    
    if user_id not in users:
        user = User(user_id, model)
        users[user_id] = user.__dict__()[user.user_id]
    else:
        if "llama3.3" in model or "gpt-4o" in model:
            users[user_id]["user_model"] = model
        else:
            client.chat_postMessage(
                channel=channel_id,
                text="Invalid model. Please choose between Llama3.3 and GPT-4o."
            )
            return
        
    with open(USER_JSON_PATH, "w") as f:
        json.dump(users, f)
    
    
    client.chat_postMessage(
        channel=channel_id,
        text=f"Hi there, <@{user_id}>! You've chosen the {model} model. I'll use this model for future responses."
    )



    
    
@app.command("/process_pdf")
def handle_process_pdf(ack, body, client):

    ack()
    user_id = body["user_id"]
    channel_id = body["channel_id"]
    questions[user_id] = body["text"] 
    

    waiting_for_file[user_id] = channel_id
    client.chat_postMessage(
        channel=channel_id,
        text="Please upload the PDF file you'd like to process.",
        mrkdwn=True

    )
    
@app.event("app_mention")
def handle_app_mention(event, client):
    user = event["user"]
    user_text = event["text"]
    channel_id = event["channel"]
    generated_script = create_empty_sh_file(GENERATED_SCRIPTS_PATH+f"app_mention_{user}{count_files()}.sh")
    script_path = SCRIPTS_PATH+"submit_app_mention.sh"
    args = [script_path, user_text, user, channel_id, generated_script]
    try:
        subprocess.run(args, check=True)
        print("Script executed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing the script: {e}")
        print(f"stderr: {e.stderr}")
        print(f"stdout: {e.stdout}")
    


@app.event("file_shared")
def handle_file_shared(event, client):
    user_id = event.get("user_id")
    file_id = event.get("file_id")
    
    if user_id in waiting_for_podcast:
        channel_id = waiting_for_podcast[user_id]
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
                generated_script = create_empty_sh_file(GENERATED_SCRIPTS_PATH+f"generate_podcast_{user_id}{count_files()}.sh")
                args = [script_path, PAPERS_PATH + file_name, user_id, channel_id, "Here's the audio podcast for your pdf!", generated_script]

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
            print(f"Error processing file: {e}")
            client.chat_postMessage(
                channel=channel_id,
                text=f"An error occurred while processing your file. Please try again. {e}"
            )
        


    
    elif user_id in waiting_for_file:
        channel_id = waiting_for_file[user_id]
        try: 
            user_questions = questions[user_id]
            del questions[user_id]
        except: user_questions = ""
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
                    user = User(user_id, "llama3.3")
                    users[user_id] = user.__dict__()[user.user_id]
                    with open(USER_JSON_PATH, "w") as f:
                        json.dump(users, f)
                
                
                model = users[user_id]["user_model"]
                
                script_path = SCRIPTS_PATH+"submit_process_pdf.sh"
                generated_script = create_empty_sh_file(GENERATED_SCRIPTS_PATH+f"process_pdf_{user_id}{count_files()}.sh")
                args = [script_path, model, PAPERS_PATH+file_name,user_questions,user_id, channel_id, generated_script]
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
            print(f"Error processing file: {e}")
            client.chat_postMessage(
                channel=channel_id,
                text=f"An error occurred while processing your file. Please try again. {e}"
            )
    else:
        pass  

if __name__ == "__main__":
    handler = SocketModeHandler(app, APP_TOKEN)
    handler.start()
