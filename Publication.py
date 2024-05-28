from scholarly import scholarly
import json
from pprint import pp

class Publication:
    def __init__(self,publication_filled,author_name) -> None:
        self.publication_filled = publication_filled
        self.author_name = author_name
        
        self.title = self.get_publication_title()
        self.abstract = self.get_publication_abstract()
        self.author = self.get_author_name()
        self.year = self.get_year()
        self.topic = self.get_topic() # Maybe to edit ??
        
    def get_topic(self):
        try:
            with open('output.json','r') as file:
                data = json.load(file)
            return data[self.author]['topic']
        except Exception as e:
            return self.__get_topic('response.json')
    
    def get_publication_title(self):
        return self.publication_filled['bib']['title'] 

    def get_publication_abstract(self):
        return self.publication_filled['bib']['abstract']

    def get_author_name(self):
        return self.publication_filled['bib']['author']

    def get_year(self):
        return self.publication_filled['bib']['pub_year']
    
    def __get_topic(self, file):
        
        topics = []
        with open(file, 'r') as file:
            data = json.load(file)
        
        for category, item in data.items():
            for keyword in item['keywords']:
                for word in self.abstract.split():
                    if word.lower() in keyword:
                        topics.append(category)
                for word in self.title.split():
                    if word.lower() in keyword:
                        topics.append(category)
                    
        return topics
    
    
            
        
    
        

