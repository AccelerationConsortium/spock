from scholarly import scholarly
import json
from pprint import pp


# Thème + mots clées en json + 
def get_last_publication(author_name: str):
    search_query = scholarly.search_author(author_name)
    first_author_result = next(search_query)
    author = scholarly.fill(first_author_result )
    first_publication = sorted(author['publications'], key= lambda x: int(x['bib']['pub_year']) if 'pub_year' in x['bib'] else 0, reverse=True)[0]
    first_publication_filled = scholarly.fill(first_publication)
    return first_publication_filled

def get_publication_titles(author_filled):
    return author_filled['bib']['title'] 

def get_publication_abstract(author_filled):
    return author_filled['bib']['abstract']

def get_topics(author_filled):
    # Load the json file
    with open('response.json', 'r') as file:
        data = json.load(file)
    for category, item in data.items():
        added = False
        for word in get_publication_titles(author_filled).split():
            if word.lower() in item['keywords'] and not added:
                data[category]['articles'].append(get_publication_titles(author_filled)) 
                added = True
        for word in get_publication_abstract(author_filled).split():
            if word.lower() in item['keywords'] and not added:
                data[category]['articles'].append(get_publication_titles(author_filled))
                added = True

    # Writing 
    with open('response.json', 'w') as file:
        for category in data:
            data[category]['articles'] = list(set(data[category]['articles']))
        json.dump(data, file)
    
    
                    
    
        
    
    
        
