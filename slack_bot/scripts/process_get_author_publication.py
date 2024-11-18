## TO EDIT


import argparse
import os
import json
from slack_sdk import WebClient
from spock_literature import Spock

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
    spock = Spock(model=model, paper=paper_path, custom_questions=user_questions)
    spock()
    response_output = spock.format_output()

    # Save the analyzed paper
    analyzed_papers_path = "/home/m/mehrad/brikiyou/scratch/spock_package/spock/slack_bot/analyzed_publications.json"
    with open(analyzed_papers_path, "r") as f:
        analyzed_papers = json.load(f)
    analyzed_papers[os.path.basename(paper_path)] = response_output
    with open(analyzed_papers_path, "w") as f:
        json.dump(analyzed_papers, f)

    # Send the response back to the user via Slack
    BOT_TOKEN = os.getenv('BOT_TOKEN')
    client = WebClient(token=BOT_TOKEN)
    client.chat_postMessage(
        channel=channel_id,
        text=f"Hi there, <@{user_id}>! Your file `{os.path.basename(paper_path)}` has been processed. \n {response_output}",
        mrkdwn=True
    )

if __name__ == "__main__":
    main()
