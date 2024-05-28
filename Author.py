from scholarly import scholarly
import json
from Publication import Publication
from pprint import pp

class Author:
    def __init__(self,author) -> None:
        self.author_name = author
        self.last_publication = Publication(self.get_last_publication())



    def get_last_publication(self):
        search_query = scholarly.search_author(self.author_name)
        first_author_result = next(search_query)
        author = scholarly.fill(first_author_result )
        first_publication = sorted(author['publications'], key= lambda x: int(x['bib']['pub_year']) if 'pub_year' in x['bib'] else 0, reverse=True)[0]
        first_publication_filled = scholarly.fill(first_publication)
        return first_publication_filled



    def write_abstract(author_filled, author):
        with open('output.json','r') as file:
            data = json.load(file)
        data[author] = {"title": get_publication_titles(author_filled),"abstract": get_publication_abstract(author_filled), "topic": [], "author": get_author_name(author_filled), "year": get_year(author_filled)}
        with open('output.json','w') as file:
            json.dump(data, file)


    def get_topics(author_filled,author):
        
        with open('response.json', 'r') as file:
            deta = json.load(file)
            
        with open('output.json', 'r') as file:
            output = json.load(file)
            
        for category, item in deta.items():
            for keyword in item['keywords']:
                if keyword in get_publication_abstract(author_filled).lower():
                    output[author]['topic'].append(category)
                if keyword in get_publication_titles(author_filled).lower():
                    output[author]['topic'].append(category)
    
    """
    # Load the json file
    with open('response.json', 'r') as file:
        data = json.load(file)
    
    for category, item in data.items():
        
        for keyword in item['keywords']:
            if keyword in get_publication_abstract(author_filled).lower():
                data[category]['articles'].append(get_publication_titles(author_filled)) 
            
            # if keyword in get_publication_titles(author_filled).lower():
            #     data[category]['articles'].append(get_publication_titles(author_filled))
    for word in get_publication_titles(author_filled).split():
        for category, item in data.items():
            for keyword in item['keywords']:
                if keyword in word.lower():
                    data[category]['articles'].append(get_publication_titles(author_filled))
    for word in get_publication_abstract(author_filled).split():
        for category, item in data.items():
            for keyword in item['keywords']:
                if keyword in word.lower():
                    data[category]['articles'].append(get_publication_titles(author_filled))
    """

    # Writing 
    with open('output.json', 'w') as file:
        for article in output:
            output[article]['topic'] = list(set(output[article]['topic']))
        # print(output)
        json.dump(output, file)
        
    
    
        
