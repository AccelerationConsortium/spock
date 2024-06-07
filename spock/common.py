"""The common module contains common functions and classes used by the other modules.
"""

import concurrent.futures
from author import Author
from publication import Publication


def setup_json(author):
    try:
        author = author[:-1]
        author_filled = Author(author)
        author_filled.setup_author('json/ouput.json')
        print(f"Topics for {author} have been updated")
    except Exception as e:
        print(f"Couldn't find the google scholar profile for {author}: {e}")

def setup() -> None:
    with open("authors.txt","r") as file:
        authors = file.readlines()
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:  # Adjust max_workers as needed
        executor.map(setup_json, authors)


def process_scholar(scholar):
    key = scholar[0]
    value = scholar[1]
    try:

        author = Author(key)
        #print(f'value title= {value["title"]} \n author title = {author.get_last_publication()["bib"]["title"]}')
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
                channel=channel_id, 
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
            client.chat_postMessage(
                channel=channel_id,
                text="Set up is complete"
            )

        except SlackApiError as e:
            print(f"Error posting message: {e.response['error']}")
    
def handle_socket_mode_request(client: SocketModeClient, req: SocketModeRequest):
    if req.type == "slash_commands":
        process_slash_command(req.payload)
        client.send_socket_mode_response(SocketModeResponse(envelope_id=req.envelope_id))

 
