import argparse
import os
import json
from slack_sdk import WebClient
from spock_literature import Spock
from langchain_community.callbacks import get_openai_callback


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
    

    # Prepare custom questions
    if questions_str:
        user_questions = questions_str.split("||")
    else:
        user_questions = []

    # Perform the processing
    
    with get_openai_callback() as cb:
        spock = Spock(model=model, paper=paper_path, custom_questions=user_questions)
        spock()
        response_output = spock.format_output()
        cost = cb.total_cost


    # Send the response back to the user via Slack
    BOT_TOKEN = os.getenv('BOT_TOKEN')
    client = WebClient(token=BOT_TOKEN)
    client.chat_postMessage(
        channel=channel_id,
        text=f"Hi there, <@{user_id}>! Your file `{os.path.basename(paper_path)}` has been processed. \n {response_output} \n Cost (USD): {cost}",
        mrkdwn=True
    )

if __name__ == "__main__":
    main()
