from article_topics import *

with open("authors.txt","r") as file:
    authors = file.readlines()
    for author in authors:
        try:
            author_filled = get_last_publication(author)
            get_topics(author_filled)
            print(f"Topics for {author} have been updated")
        except Exception as e:
            print(f"Couldn't find the google scholar profile for {author}: {e}")
