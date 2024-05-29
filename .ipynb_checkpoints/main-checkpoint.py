from Classes.Author import Author
from Classes.Publication import Publication


with open("authors.txt","r") as file:
    authors = file.readlines()
    for author in authors:
        try:
            author = author[:-1]
            author_filled = Author(author)
            author_filled.setup_author('json/ouput.json')
            print(f"Topics for {author} have been updated")
        except Exception as e:
            print(f"Couldn't find the google scholar profile for {author}: {e}")
