from scholarly import scholarly
from spock_literature.utils.Publication_scholarly import Publication_scholarly as Publication
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

    def get_last_publication(self, count) -> dict:
        """
        Get the last publication of the author.

        Returns:
            dict: A dict containing information about the last publication.
        """
        try:
            publications_filled = []
            search_query = scholarly.search_author(self.author_name)
            first_author_result = next(search_query)
            author = scholarly.fill(first_author_result)
            publications = sorted(author['publications'], 
                                    key=lambda x: int(x['bib']['pub_year'])
                                    if 'pub_year' in x['bib'] else 0, 
                                    reverse=True)[0:count]
            for publication in publications:
                publications_filled.append(scholarly.fill(publication))
            return publications_filled
        except Exception as e:
            print(f"An error occurred, couldnt get the latest publications: {e}")

    def __call__(self,count:int=1):
        """
        Setup the author by adding their last publication to a JSON file.

        Args:
            output_file (str): The path to the JSON file.

        Returns:
            data (dict): A dict containing the author's last publication.
        """
        
        data = {}
        author_publications = self.get_last_publication(count)
        for publication in author_publications:
            publication = Publication(publication)
            publication_data = {
                "title": publication.title,
                "abstract": publication.abstract,
                "author": publication.author, 
                "year": publication.year,
                "url": publication.url,
            }
            if self.author_name in data:
                data[self.author_name].append(publication_data)
            else:
                data[self.author_name] = [publication_data]
        return data
    
    
if __name__ == "__main__":
    author = Author("Mehrad Ansari")
    print(author(2))