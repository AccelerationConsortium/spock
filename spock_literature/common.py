"""The common module contains common functions and classes used by the other modules.
"""

try:
    from .author import Author
    from .publication import Publication
    from .bot import Bot
except:
    from author import Author
    from publication import Publication
    from bot import Bot
    

def setup_json(author):
    import concurrent.futures

    try:
        author = author[:-1]
        author_filled = Author(author)
        print('Author created successfully for ' + author)
        author_filled.setup_author('json/output.json')
        print(f"Topics for {author} have been updated")
    except Exception as e:
        print(f"Couldn't find the google scholar profile for {author}: {e}")

def setup() -> None:
    import concurrent.futures
    with open("data/authors.txt","r") as file:
        authors = file.readlines()
        print(authors)
    #with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:  # Adjust max_workers as needed
        #executor.map(setup_json, authors)
    for author in authors:
        setup_json(author)



def download_pdfs():
    import json
    
    with open("json/output.json", 'r') as file:
        data = json.load(file)
    for author in data:
        if data[author]['url'] != None:
            try:
                pub = Publication(Author(author).get_last_publication())
                pub.download_pdf(f"pdfs/")
                print(f"PDF for {author} downloaded successfully")
            except Exception as e:
                print(f"Couldn't download the pdf for {author}: {e}")
        else:
            print(f"No pdf found for {author}")
            
download_pdfs()

    
