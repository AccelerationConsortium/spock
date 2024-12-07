# TTILE OR DOI
import argparse
import os
import json
from slack_sdk import WebClient
from spock_literature import Spock
import re

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="gpt-4o")
    parser.add_argument('--publication', required=True)
    parser.add_argument('--questions', default="")
    parser.add_argument('--user_id', required=True)
    parser.add_argument('--channel_id', required=True)
    args = parser.parse_args()

    model = args.model
    publication = args.publication
    questions_str = args.questions
    user_id = args.user_id
    channel_id = args.channel_id
    

    # Prepare custom questions
    if questions_str:
        user_questions = questions_str.split("||")
    else:
        user_questions = []
        
    if re.match(r'^10\.\d{4,9}/[-._;()/:A-Za-z0-9]+$', publication):
        # DOI
        doi = publication
        spock = Spock(model=model, publication_doi=doi, custom_questions=user_questions)

    elif re.match(r"https?://(www\.)?[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}/[a-zA-Z0-9\-_/]+", publication):
        # URL
        url = publication
        spock = Spock(model=model, publication_url=url, custom_questions=user_questions)

    else:
        # Title
        title = publication
        spock = Spock(model=model, publication_title=title, custom_questions=user_questions)

    try:
        spock()
    except Exception as e:
        client.chat_postMessage(
        channel=channel_id,
        text=f"Hi there, <@{user_id}>! Please upload the PDF for this article, couldn't find the publication with the given title/DOI/URL",
        mrkdwn=True
    )

        return
        
    response_output = spock.format_output()

    # Save the analyzed paper
    
    """
    analyzed_papers_path = "/home/m/mehrad/brikiyou/scratch/spock_package/spock/slack_bot/analyzed_publications.json"
    with open(analyzed_papers_path, "r") as f:
        analyzed_papers = json.load(f)
    analyzed_papers[os.path.basename(paper_path)] = response_output
    with open(analyzed_papers_path, "w") as f:
        json.dump(analyzed_papers, f)
    """
    
    
    # Send the response back to the user via Slack
    BOT_TOKEN = os.getenv('BOT_TOKEN')
    client = WebClient(token=BOT_TOKEN)
    client.chat_postMessage(
        channel=channel_id,
        text=f"Hi there, <@{user_id}>! Your file has been processed. \n {response_output}",
        mrkdwn=True
    )

if __name__ == "__main__":
    main()
