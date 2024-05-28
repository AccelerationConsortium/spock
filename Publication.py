from scholarly import scholarly
import json
from pprint import pp

class Publication:
    def __init__(self,publication_filled) -> None:
        self.publication_filled = publication_filled
        self.title = self.get_publication_title()
        self.abstract = self.get_publication_abstract()
        self.author = self.get_author_name()
        self.year = self.get_year()
    
    def get_publication_title(self):
        return self.publication_filled['bib']['title'] 

    def get_publication_abstract(self):
        return self.publication_filled['bib']['abstract']

    def get_author_name(self):
        return self.publication_filled['bib']['author']

    def get_year(self):
        return self.publication_filled['bib']['pub_year']

