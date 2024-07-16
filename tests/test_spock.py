
"""Tests for `spock` package."""

from scholarly import scholarly
import pytest
from spock_literature.author import Author
from bs4 import BeautifulSoup



names = [
    "Mehrad Ansari",
    "Jay Werber",
    "Jason Hattrick-Simpers",
    "Cheryl Arrowsmith",
    "Robert Batey",
    "Christine Allen, Toronto",
    "Mohamad Moosavi, Toronto",
    "Tejs Vegge",
    "Tonio Buonassisi"
]


def get_manually_latest_article(name:str):
     import requests
     search_query = scholarly.search_author(name)
     scholar_id = next(search_query)['scholar_id']
     url = f"https://scholar.google.com/citations?hl=en&user={scholar_id}&view_op=list_works&sortby=pubdate"
     print(url)
     response = requests.get(url)
     print(response.status_code)
     if response.status_code == 200:
          html_content = response
          soup = BeautifulSoup(html_content, 'html.parser')
          tbody = soup.find('tbody', id='gsc_a_b')
          first_tr = tbody.find('tr')

          title = first_tr.find('a', class_='gsc_a_at').text

          authors = first_tr.find_all('div', class_='gs_gray')[0].text

          journal_info = first_tr.find_all('div', class_='gs_gray')[1].text
          journal, year = journal_info.rsplit(',', 1)
          year = year.strip() 
          return {"title":title, "authors":authors, "journal":journal, "year":year}

          # print(f"Title: {title}")
          # print(f"Authors: {authors}")
          # print(f"Journal: {journal}")
          # print(f"Year: {year}")
     else:
          print(f"Failed to fetch the page. Status code: {response.status_code}")
          return None

get_manually_latest_article('Mehrad Ansari')
@pytest.fixture(params=names)
def create_valid_author(request):
    return Author(request.param)

@pytest.fixture
def get_latest_article():
    def _get_latest_article(author: Author):
        return author.get_last_publication()
    return _get_latest_article

def test_create_valid_author(create_valid_author, get_latest_article):
    author = create_valid_author
    data_manually = get_manually_latest_article(author.author_name)
    try:
          data_get_latest_article = get_latest_article(author)
          has_article = True
    except:
          has_article = False
    assert author.author_name in names
    if data_manually is not None and has_article: # In case of 429 error
        assert data_manually['title'] == data_get_latest_article['bib']['title']
        assert data_manually['authors'] == data_get_latest_article['bib']['author']
        assert data_manually['journal'] == data_get_latest_article['bib']['journal']
    else:
         print(f"No data found for the author {author.author_name}")

    print(f"Test passed for {author.author_name}")
     

@pytest.fixture
def create_false_author():
     return Author('Definitly a false name like no-one would have a name like this :/')

def test_false_author(create_false_author):
     with pytest.raises(StopIteration):
        create_false_author.get_last_publication()


"""
query = scholarly.search_pubs("History-agnostic battery degradation inference")
pub = next(query)
print(scholarly.bibtex(pub))

import pytest
from unittest.mock import patch, MagicMock
from spock_literature.publication import Publication  # Replace 'your_module' with the actual name of your module


@pytest.fixture
def sample_publication(title):
     query = scholarly.search_pubs(title)
     pub = next(query)
     return pub

@pytest.fixture
def publication(sample_publication):
    return Publication(sample_publication)

def test_publication_initialization(publication, sample_publication):
    assert publication.title == sample_publication['bib']['title']
    assert publication.abstract == sample_publication['bib']['abstract'].lower()
    assert publication.author == sample_publication['bib']['author']
    assert publication.year == sample_publication['bib']['pub_year']
    

"""


from spock_literature.bot import Bot_LLM
from langchain_community.llms import Ollama



# Test the topic 
@pytest.fixture
def get_topic_publication_abstract():
    def _get_topic_publication_abstract(abstract:str, input_file:str):
        return Bot_LLM().get_topic_publication_abstract(abstract, input_file)
    return _get_topic_publication_abstract
def test_bot_llm():
    bot = Bot_LLM()



# Test tge