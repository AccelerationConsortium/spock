from Classes.Author import Author
from Classes.Publication import Publication
import time
import concurrent.futures
import json
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import concurrent.futures
from slack_sdk import WebClient
from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse

# TODO:
# Finish the bot and the /commands
# Clean the code





# author = Author('Mehrad Ansari')
# print(author.get_last_publication())

# Your Slack bot token
slack_token = 'xoxb-BOT'

# Initialize a Web API client
client = WebClient(token=slack_token)

# The channel ID of the private channel you want to send the message to
channel_id = 'C072YU8S539'

# The message you want to send

socket_mode_client = SocketModeClient(app_token="xapp-1-BOT", web_client=client)


# Initial Set-up
def setup() -> None:
    with open("authors.txt","r") as file:
        authors = file.readlines()
        for author in authors:
            try:
                author = author[:-1]
                author_filled = Author(author)
                author_filled.setup_author('json/ouput.json')
                print(f"Topics for {author} have been updated")
            except Exception as e:
                print(f"Couldn't find the google scholar profile for {author}: {e}")



def process_scholar(scholar):
    for key,value in scholar:
        try:
            author = Author(key)
            print(f'value title= {value["title"]} \n author title = {author.get_last_publication()["bib"]["title"]}')
            if value['title'] != author.get_last_publication()['bib']['title']:
                
                print(f"Updating topics for {author}")
                
                try:
                    last_publication = Publication(author.get_last_publication())
                except Exception as e:
                    print(f"Couldn't fetch the last publication for {author}: {e}")
                    
                
                text_message = f":rolled_up_newspaper::test_tube: {author.author_name} has an update on Google Scholar!\n\
                        ```Title: {last_publication.title}\nCitation: {last_publication.citation}\nYear: {last_publication.year}```"
                try:
                    response = client.chat_postMessage(
                    channel="CHANNEL_ID", 
                    text=text_message)
                except Exception as e:
                    print(f"Couldn't send the message to slack: {e}")
                
                # Updating the Json file
                try:
                    author.setup_author('json/ouput.json')
                except Exception as e:
                    print(f"Couldn't Overwrite the old data for: {author}: {e}")

            
            print(f"Topics for {author} have been updated")
        except Exception as e:
            print(f"Couldn't find the google scholar profile for {author}: {e}")

def process_slash_command(payload):
    command = payload['command']
    user_id = payload['user_id']
    text = payload['text']
    channel_id = payload['channel_id']

    if command == '/hello':
        response_message = f"Hello <@{user_id}>!"

        try:
            # Post the message
            client.chat_postMessage(
                channel=channel_id,
                text=response_message
            )
            print("/hello was successfully posted")
        except SlackApiError as e:
            print(f"Error posting message: {e.response['error']}")
            
    elif command == '/setup':
        response_message = f"Hello <@{user_id}>! It's loading Data, it might take some time"
        try:
            # Post the message
            client.chat_postMessage(
                channel=channel_id,
                text=response_message
            )
            print("/setup was successfully posted")
            setup()
        except SlackApiError as e:
            print(f"Error posting message: {e.response['error']}")
    
        

# Function to handle incoming Socket Mode requests
def handle_socket_mode_request(client: SocketModeClient, req: SocketModeRequest):
    if req.type == "slash_commands":
        process_slash_command(req.payload)
        client.send_socket_mode_response(SocketModeResponse(envelope_id=req.envelope_id))

# Register the handler to the client
socket_mode_client.socket_mode_request_listeners.append(handle_socket_mode_request)

if __name__ == "__main__":
    socket_mode_client.connect()
    while True:
        
        with open('json/ouput.json', 'r') as file:
            scholars_publications = json.load(file)

        #with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:  # Adjust max_workers as needed
            #executor.map(process_scholar, scholars_publications.items())
        
        map(process_scholar, scholars_publications.items())

        print('Waiting!')
        time.sleep(1)

"""
while True:
    
    with open('json/ouput.json', 'r') as file:
        scholars_publications = json.load(file)

    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:  # Adjust max_workers as needed
        executor.map(process_scholar, scholars_publications.items())

    print('Waiting!')
    time.sleep(900)
    
"""