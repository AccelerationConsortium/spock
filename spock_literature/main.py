from common import *
import json
import time
import concurrent.futures
from author import Author


if __name__ == "__main__":
    with open('json/output.json','r') as file:
        data = json.load(file)
        
    for author in data:
        try:
            author_filled = Author(author)
            print('Author created successfully for ' + author)
            author_filled.setup_author('json/output.json')
            print(f"Topics for {author} have been updated")
        except Exception as e:
            print(f"Couldn't find the google scholar profile for {author}: {e}")
        
        