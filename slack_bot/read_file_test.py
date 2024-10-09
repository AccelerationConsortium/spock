import os
import requests
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
#from  ../spock_



BOT_TOKEN = "xoxb-1089129130001-7130503874147-1eJXyg9HdahYaxOHANd8iyc0"
APP_TOKEN = "xapp-1-A074H63F6EL-7837148342518-b5be6962fadd66118e88e8aa2c1cdca5f393404870ef5d49ee25d944fc6fc02b"

app = App(token=BOT_TOKEN)
waiting_for_file = {}


@app.command("/help")
def help(ack, body, client):
    ack()
    user_id = body["user_id"]
    channel_id = body["channel_id"]
    client.chat_postMessage(
        channel=channel_id,
        text="I am a bot that can process PDF files. To get started, type `/processpdf`."
    )
    
    

@app.command("/processpdf")
def handle_processpdf(ack, body, client):

    ack()
    user_id = body["user_id"]
    channel_id = body["channel_id"]

    waiting_for_file[user_id] = channel_id
    client.chat_postMessage(
        channel=channel_id,
        text="Please upload the PDF file you'd like to process."
    )
    
@app.command("/hello")
def hello_command(ack, body):
    print("hello")
    user_id = body["user_id"]
    ack(f"Hi, <@{user_id}>!")

@app.event("app_mention")
def handle_app_mention(event, say):
    user = event["user"]
    say(f"Hi there, <@{user}>! You mentioned me.")
    # Add LLM
    


@app.event("file_shared")
def handle_file_shared(event, client, logger):
    user_id = event.get("user_id")
    file_id = event.get("file_id")

    # Check if the user is in the waiting list
    if user_id in waiting_for_file:
        channel_id = waiting_for_file[user_id]

        # Remove the user from the waiting list
        del waiting_for_file[user_id]

        # Get file info
        try:
            file_info_response = client.files_info(file=file_id)
            file_info = file_info_response["file"]
            if file_info["filetype"] == "pdf":
                url_private = file_info["url_private_download"]
                file_name = file_info["name"]

                # Download the file using the bot token for authentication
                response = requests.get(url_private, headers={'Authorization': f'Bearer {BOT_TOKEN}'})
                with open(file_name, "wb") as f:
                    f.write(response.content)

                # Process the PDF file as needed
                # For example, extract text, analyze content, etc.
                # ...
                
                

                # Notify the user
                client.chat_postMessage(
                    channel=channel_id,
                    text=f"Your file `{file_name}` has been processed."
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
