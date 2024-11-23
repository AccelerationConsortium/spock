import argparse
import os
import json
from slack_sdk import WebClient
from spock_literature import Spock
import re
from spock_literature.utils.Author import Author

BOT_TOKEN = os.getenv('BOT_TOKEN')
client = WebClient(token=BOT_TOKEN)
def upload_audio_file(channel_id, file_path, initial_comment):
    try:
        response = client.files_upload_v2(
            channel=channel_id,
            initial_comment=initial_comment,
            file=file_path,
        )
        if response.get('ok'):
            print('File uploaded successfully!')
        else:
            print(f"Failed to upload file: {response.get('error')}")
    except Exception as e:
        print(f"An error occurred: {e}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paper', required=True)
    parser.add_argument('--audio_file_path', required=True)
    parser.add_argument('--user_id', required=False)
    parser.add_argument('--channel_id', required=True)
    parser.add_argument('--initial_comment', default="")
    parser.add_argument('--add_transcript', default=False)
    args = parser.parse_args()

    paper = args.paper
    user_id = args.user_id
    channel_id = args.channel_id
    initial_comment = args.initial_comment
    add_transcript = args.add_transcript
    
    spock = Spock(paper=paper)
    audio_file_path, transcript = spock.generate_podcast()
    upload_audio_file(channel_id, audio_file_path, initial_comment)
    if add_transcript:
        client.chat_postMessage(
            channel=channel_id,
            text=f"Here is the transcript of the podcast:\n {transcript}",
            mrkdwn=True
        )

        
        
        
if __name__ == "__main__":
    main()

