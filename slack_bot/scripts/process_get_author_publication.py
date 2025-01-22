import argparse
import os
from slack_sdk import WebClient
from spock_literature import Spock
from spock_literature.utils.Author import Author

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--author', required=True)
    parser.add_argument('--count', required=True)
    parser.add_argument('--user_id', required=True)
    parser.add_argument('--channel_id', required=True)
    args = parser.parse_args()

    author = args.author
    count = args.count
    user_id = args.user_id
    channel_id = args.channel_id
    
    
    author = Author(author)
    publications = author(int(count))
    
    output = ""
    for author, works in publications.items():
        for work in works:
            output += f"📄 Title: {work['title']}\n"
            output += "📜 Abstract:\n"
            output += f"    {work['abstract']}\n"
            output += f"✍️ Authors: {work['author']}\n"
            output += f"📅 Year: {work['year']}\n"
            output += f"🔗 URL: {work['url']}\n"
            output += "\n" + "-"*50 + "\n\n"

    
    
    BOT_TOKEN = os.getenv('BOT_TOKEN')
    client = WebClient(token=BOT_TOKEN)
    client.chat_postMessage(
        channel=channel_id,
        text=f"Hi there, <@{user_id}>! Here are the {count} latest publications of {author}: {output}",
        mrkdwn=True
    )
    
if __name__ == "__main__":
    main()
