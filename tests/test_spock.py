"""
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from .fake_llm import FakeChatModel
from spock_literature.spock import Spock
from spock_literature.utils.Url_downloader import URLDownloader
from langchain_ollama import OllamaLLM

@pytest.fixture
def spock_instance(tmp_path):
    return Spock(
        model="llama3.3",
        paper=None,
        custom_questions=["What is the main novelty?"],
        publication_doi="10.1234/example.doi",
        publication_title="Sample Title",
        publication_url="http://example.com",
        papers_download_path=str(tmp_path) + "/"
    )

def test_spock_initialization(spock_instance):
    s = spock_instance
    assert isinstance(s.llm, OllamaLLM())
    assert s.custom_questions == ["What is the main novelty?"]
    assert s.publication_doi == "10.1234/example.doi" 
    assert s.publication_title == "Sample Title"
    assert s.publication_url == "http://example.com"
    assert s.paper is None

def test_spock_download_pdf(spock_instance):
    responses = [""]
    fake_llm = FakeChatModel() 

def test_spock_format_output():
    s = Spock(model="llama3.3")
    s.paper_summary = "Paper summary"
    s.topics = "Topic1/Topic2"
    s.questions = {
        "Q1": {"question": "Is this a test?", "output": {"response": "Yes", "sentence": "This is a test sentence."}}
    }
    output = s.format_output()
    assert "Paper summary" in output
    assert "Topic1/Topic2" in output
    assert "Q1" in output
    assert "Yes" in output
    assert "This is a test sentence." in output
"""