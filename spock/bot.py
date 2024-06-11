from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_sdk import WebClient
from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse
from common import setup

class Bot:
    def __init__(self, slack_bot_token:str, slack_app_token:str, channel_id:str):
        self.slack_bot_token = slack_bot_token
        self.client = WebClient(token=self.slack_bot_token)
        self.channel_id = channel_id
        self.slack_app_token = slack_app_token
        self.socket_mode_client = SocketModeClient(app_token=self.slack_app_token)
        self.register_handler_to_client()
        
    def process_slash_command(self,payload):
        command = payload['command']
        user_id = payload['user_id']
        text = payload['text']
        channel_id = payload['channel_id']

        if command == '/hello':
            response_message = f"Hello <@{user_id}>!"

            try:
                # Post the message
                self.client.chat_postMessage(
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
                self.client.chat_postMessage(
                    channel=channel_id,
                    text=response_message
                )
                print("/setup was successfully posted")
                setup() # This function is not defined yet
                self.client.chat_postMessage(
                    channel=channel_id,
                    text="Set up is complete"
                )

            except SlackApiError as e:
                print(f"Error posting message: {e.response['error']}")

    def handle_socket_mode_request(self, req: SocketModeRequest):
        if req.type == "slash_commands":
            self.process_slash_command(req.payload)
            self.socket_mode_client.send_socket_mode_response(SocketModeResponse(envelope_id=req.envelope_id))
        
    def register_handler_to_client(self):
        self.socket_mode_client.socket_mode_request_listeners.append(self.handle_socket_mode_request)    

                
        