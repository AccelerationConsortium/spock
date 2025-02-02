import argparse
import os
from slack_sdk import WebClient
from spock_literature import Spock
from langchain_community.callbacks import get_openai_callback
import time
import json


USERS_FILE = "/home/m/mehrad/brikiyou/scratch/spock/slack_bot/users.json"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--paper', required=True)
    parser.add_argument('--questions', default="")
    parser.add_argument('--user_id', required=True)
    parser.add_argument('--channel_id', required=True)
    args = parser.parse_args()

    model = args.model
    paper_path = args.paper
    questions_str = args.questions
    user_id = args.user_id
    channel_id = args.channel_id
    
    with open(USERS_FILE, "r") as f:
        users = json.load(f)
        
    if user_id not in users:
        client.chat_postMessage(
            channel=channel_id,
            text=f"Hi there, <@{user_id}>! An error occured, couldn't find your user information.",
            mrkdwn=True
        )
        raise ValueError(f"User {user_id} not found in the users file.")
        
    

    if questions_str:
        print("Custom questions provided: ", questions_str)
        user_questions = questions_str.split("||")
    else:
        user_questions = []

    start_time = time.time()
    with get_openai_callback() as cb:
        spock = Spock(model=model, paper=paper_path, custom_questions=user_questions, settings=users[user_id]["settings"])
        spock()
        response_output = spock.format_output()
        cost = round(cb.total_cost,2)
    total_time = round(time.time() - start_time,2)
    


    BOT_TOKEN = os.getenv('BOT_TOKEN')
    client = WebClient(token=BOT_TOKEN)
    client.chat_postMessage(
        channel=channel_id,
        text=f"Hi there, <@{user_id}>! Your file `{os.path.basename(paper_path)}` has been processed. \n {response_output} \n Cost (USD): {cost} \n Time taken: {total_time} seconds",
        mrkdwn=True
    )

if __name__ == "__main__":
    main()
