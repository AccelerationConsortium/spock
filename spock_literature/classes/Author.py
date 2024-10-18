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

    def __call__(self, output_file,publication:Publication=None):
        """
        Setup the author by adding their last publication to a JSON file.

        Args:
            output_file (str): The path to the JSON file.

        Returns:
            None
        """
        with open(output_file, 'r') as file:
            data = json.load(file)
        author_last_publication = Publication(self.get_last_publication()) if publication != None else publication
        

        
        data[self.author_name] = {
            "title": author_last_publication.title,
            "abstract": author_last_publication.abstract,
            "author": author_last_publication.author, 
            "year": author_last_publication.year,
            "url": author_last_publication.url,
            "pdf": author_last_publication.pdf,
        }
        
        
        with open(output_file, 'w') as file:
            json.dump(data, file)
