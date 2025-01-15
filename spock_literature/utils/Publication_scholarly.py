class Publication_scholarly(): 
    def __init__(self,publication_filled, get_topic:bool=False) -> None:
        """Initialize a Publication object."""
        self.publication_filled = publication_filled
        self.title = self.get_publication_title()
        self.abstract = self.get_publication_abstract().lower()
        self.author = self.get_author_name()
        self.year = self.get_year()
        self.url = self.get_publication_url()
        self.citation = self.get_citation()
        self.topic = self.get_topic() if get_topic else None
    
      
    def get_publication_url(self) -> str:
        return self.publication_filled['pub_url']
    
    def get_publication_title(self) -> str:
        return self.publication_filled['bib']['title'] 

    def get_publication_abstract(self) -> str:
        return self.publication_filled['bib']['abstract']

    def get_author_name(self) -> str:
        return self.publication_filled['bib']['author']

    def get_year(self) -> str:
        return self.publication_filled['bib']['pub_year']
    
    def get_citation(self) -> str:
        return self.publication_filled['bib']['citation']

