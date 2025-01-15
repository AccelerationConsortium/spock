# TTILE OR DOI
import argparse
import os
import json
from slack_sdk import WebClient
from spock_literature import Spock
import re
from langchain_community.callbacks import get_openai_callback
import time

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
    BOT_TOKEN = os.getenv('BOT_TOKEN')
    client = WebClient(token=BOT_TOKEN)


    # Prepare custom questions
    start_time = time.time()    
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
        with get_openai_callback() as cb:
            spock()
            cost = round(cb.total_cost,2)
            total_time = round(time.time() - start_time,2)
    except Exception as e:
        client.chat_postMessage(
        channel=channel_id,
        text=f"Hi there, <@{user_id}>! Please upload the PDF for this article, couldn't find the publication with the given title/DOI/URL",
        mrkdwn=True
    )

        return
        
    response_output = spock.format_output()
    
    client.chat_postMessage(
        channel=channel_id,
        text=f"Hi there, <@{user_id}>! Your file has been processed. \n {response_output} \n Cost (USD): {cost} \n Time taken: {total_time} seconds",
        mrkdwn=True
    )

if __name__ == "__main__":
    main()
