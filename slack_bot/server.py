import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from bot import Bot
import concurrent.futures
from slack_sdk import WebClient
from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse
from slack_bolt.adapter.socket_mode import SocketModeHandler

from slack_bolt import App

# Your Slack bot token
slack_token = 'xoxb-1089129130001-7130503874147-1eJXyg9HdahYaxOHANd8iyc0'

# Initialize a Web API client
client = WebClient(token=slack_token)

# The channel ID of the private channel you want to send the message to
# The message you want to send
message = 'File was changed'

bot = Bot('test.txt')
socket_mode_client = SocketModeClient(app_token="xapp-1-A074H63F6EL-7837148342518-b5be6962fadd66118e88e8aa2c1cdca5f393404870ef5d49ee25d944fc6fc02b", web_client=client)
app = App(token=slack_token)


@app.event("file_shared")
def file_func(payload, client, ack):
    print("okay")
    ack()

    #get the file id every time someone uploads a pdf
    my_file = payload.get('file').get('id')
    
    #get the json using files_info
    url = client.files_info(file = my_file).get('file').get('url_private')
    file_name = client.files_info(file = my_file).get('file').get('title')

    # save file
    resp = requests.get(url, headers={'Authorization': 'Bearer %s' % token})
    save_file = Path(file_name)
    save_file.write_bytes(resp.content)
@app.command("/hello")
def hello_command(ack, body):
    print("hello")
    user_id = body["user_id"]
    ack(f"Hi, <@{user_id}>!")


def process_slash_command(payload):
    global bot
    command = payload['command']
    user_id = payload['user_id']
    text = payload['text']
    channel_id = payload['channel_id']

    if command == '/lastmodified':
        response_message = f"Hello <@{user_id}>! You invoked the command with text: {text} and the latest modified time is {bot.last_modified()}"

        try:
            # Post the message
            client.chat_postMessage(
                channel=channel_id,
                text=response_message
            )
            print("/lastdate was successfully posted")
        except SlackApiError as e:
            print(f"Error posting message: {e.response['error']}")
            
    elif command == '/stop':
        response_message = f"Hello <@{user_id}>! You invoked the command with text: {text} and it's stopping the bot"
        bot.stop()
        try:
            # Post the message
            client.chat_postMessage(
                channel=channel_id,
                text=response_message
            )
            print("/stop was successfully posted")
        except SlackApiError as e:
            print(f"Error posting message: {e.response['error']}")
    elif command == '/start':
        bot.running = True
        response_message = f"Hello <@{user_id}>! You invoked the command with text: {text} and it's starting the bot"
        try:
            # Post the message
            client.chat_postMessage(
                channel=channel_id,
                text=response_message
            )
            print("/start was successfully posted")
        except SlackApiError as e:
            print(f"Error posting message: {e.response['error']}")


# Function to handle incoming Socket Mode requests
def handle_socket_mode_request(client: SocketModeClient, req: SocketModeRequest):
    if req.type == "slash_commands":
        process_slash_command(req.payload)
        client.send_socket_mode_response(SocketModeResponse(envelope_id=req.envelope_id))

# Register the handler to the client
#socket_mode_client.socket_mode_request_listeners.append(handle_socket_mode_request)

if __name__ == "__main__":
    SocketModeHandler(app, "xapp-1-A074H63F6EL-7133121371428-665e0a091a6bbdab0068fdfa9a939f66e538a35faa9cbd6db6667cf6d21c6d52" ).start()