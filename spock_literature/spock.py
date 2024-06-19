"""Main module."""
import json
import time
import concurrent.futures
from publication import Publication
from author import Author

author = Author('Mehrad Ansari')
print(author.get_last_publication())
"""
pub = Publication(author.get_last_publication())
print(pub)
print(pub.title)
print(pub.abstract)
print(pub.topic)


while True:
        
    with open('json/ouput.json', 'r') as file:
        scholars_publications = json.load(file)

    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:  # Adjust max_workers as needed
        executor.map(process_scholar, scholars_publications.items())
    

    print('Waiting!')
    time.sleep(900)
"""