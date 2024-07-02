from scholarly import scholarly
import json
from publication import Publication

class Author:
    def __init__(self, author):
        """
        Initialize an Author object.

        Args:
            author (str): The name of the author.
        """
        self.author_name = author
        
    def __repr__(self):
        """
        Return a string representation of the Author object.

        Returns:
            str: The name of the author.
        """
        return self.author_name

    def get_last_publication(self):
        """
        Get the last publication of the author.

        Returns:
            dict: A dict containing information about the last publication.
        """
        search_query = scholarly.search_author(self.author_name)
        first_author_result = next(search_query)
        author = scholarly.fill(first_author_result)
        first_publication = sorted(author['publications'], 
                                   key=lambda x: int(x['bib']['pub_year'])
                                   if 'pub_year' in x['bib'] else 0, 
                                   reverse=True)[0]
        first_publication_filled = scholarly.fill(first_publication)
        return first_publication_filled

    def setup_author(self, output_file, llm):
        """
        Setup the author by adding their last publication to a JSON file.

        Args:
            output_file (str): The path to the JSON file.

        Returns:
            None
        """
        with open(output_file, 'r') as file:
            data = json.load(file)
        author_last_publication = Publication(self.get_last_publication())
        print("publication loaded successfully")
        
        data[self.author_name] = {
            "title": author_last_publication.title,
            "abstract": author_last_publication.abstract,
            "topic": author_last_publication.get_topic(llm=llm), 
            "author": author_last_publication.author, 
            "year": author_last_publication.year,
            "url": author_last_publication.url,
            "pdf": author_last_publication.pdf,
        }
        
        with open(output_file, 'w') as file:
            json.dump(data, file)

'''
author = Author('Mehrad Ansari')
pub = Publication(author.get_last_publication())

pub.download_pdf('pdfs/')
'''