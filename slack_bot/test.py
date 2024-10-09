import os
import requests
from pathlib import Path
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# Your Slack bot token
slack_token = "xoxb-1089129130001-7130503874147-1eJXyg9HdahYaxOHANd8iyc0"
app_token = "xapp-1-A074H63F6EL-7133121371428-665e0a091a6bbdab0068fdfa9a939f66e538a35faa9cbd6db6667cf6d21c6d52"

# Initialize the Slack app
app = App(token=slack_token)

@app.event("file_shared")
def file_func(payload, client, ack):
    print("okay")
    ack()

    # Get the file ID every time someone uploads a file
    my_file = payload.get('file').get('id')

    # Get the file info
    file_info = client.files_info(file=my_file).get('file')
    url = file_info.get('url_private')
    file_name = file_info.get('title')

    # Save the file
    resp = requests.get(url, headers={'Authorization': f'Bearer {slack_token}'})
    save_file = Path(file_name)
    save_file.write_bytes(resp.content)

@app.command("/hello")
def hello_command(ack, body):
    print("hello")
    user_id = body["user_id"]
    ack(f"Hi, <@{user_id}>!")

@app.event("app_mention")
def handle_app_mention(event, say):
    user = event["user"]
    say(f"Hi there, <@{user}>!")

if __name__ == "__main__":
    SocketModeHandler(app, app_token).start()
