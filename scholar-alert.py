from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import os


BOT_TOKEN = "xoxb_YOUR_BOT_TOKEN"
app = App(token=BOT_TOKEN)


from scholarly import scholarly
from datetime import datetime

def get_latest_article(author_name):
    current_year = datetime.now().year  # Get the current year

    # Search for the author
    search_query = scholarly.search_author(author_name)

    try:
        # Fetch the author
        author = next(search_query)
        if author:
            # Fetch detailed author info
            author = scholarly.fill(author, sortby='year')
            # Filter publications for the current year
            pub = author['publications']
            publications_sorted = sorted(pub, key=lambda x: int(x['bib']['pub_year']) if 'pub_year' in x['bib'] else 0, reverse=True)

            # Find the most recent publication in the current year
            if publications_sorted:
                latest_publication = publications_sorted[0]['bib']
                return latest_publication, latest_publication['title']
            else:
                return "No publications found for the current year"
        else:
            return "Author not found"
    except StopIteration:
        return "Author not found"


def process_scholar(scholar_tuple):
    scholar, paper_title = scholar_tuple
    try:
        update, update_title = get_latest_article(scholar)
        if update_title != paper_title:
            print(update)
            name = scholar.split(',')[0]
            text_message = f":rolled_up_newspaper::test_tube: {name} has an update on Google Scholar!\n\
                            ```Title: {update['title']}\nCitation: {update['citation']}\nYear: {update['pub_year']}```"
            response = client.chat_postMessage(
                channel="CHANNEL_ID", 
                text=text_message
            )
            return scholar, update_title
    except Exception as e:
        print(f"\nCouldn't find the google scholar profile for {scholar}: {e}\n")
    return scholar, paper_title

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import os

slack_token = "xoxb-YOUR_SLACK_TOKEN"
client = WebClient(token=slack_token)
import concurrent.futures
import time
import json

while True:
    with open('scholars_publications.json', 'r') as file:
        scholars_publications = json.load(file)

    # Process in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:  # Adjust max_workers as needed
        results = executor.map(process_scholar, scholars_publications.items())

    # Update the scholars_publications with any new titles
    for scholar, new_title in results:
        scholars_publications[scholar] = new_title

    with open('scholars_publications.json', 'w') as file:
        json.dump(scholars_publications, file, indent=4)

    print('Waiting!')
    time.sleep(900) 