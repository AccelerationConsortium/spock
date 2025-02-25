import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from .fake_llm import FakeChatModel
from spock_literature.spock import Spock
from spock_literature.utils.Url_downloader import URLDownloader
from langchain_ollama import OllamaLLM
from langchain.schema import Document


def test_scan_pdf():
    mock_responses = [
        "Yes/We found mention in section 2",  
        "No/None"                           
    ]
    fake_llm = FakeChatModel(mock_responses)

    spock = Spock(paper="fake-paper.pdf", embed_model=True)
    spock.llm = fake_llm
    spock.questions = {
        "TopicA": {
            "question": "Is TopicA present?",
            "output": {"response": "", "sentence": ""}
        },
        "TopicB": {
            "question": "Is TopicB present?",
            "output": {"response": "", "sentence": ""}
        },
    }

    with patch.object(spock, "chunk_indexing", return_value=None):
        with patch.object(spock, "query_rag", side_effect=mock_responses):
            spock.scan_pdf()

    assert spock.questions["TopicA"]["output"]["response"] == "Yes"
    assert spock.questions["TopicA"]["output"]["sentence"] == "We found mention in section 2"
    assert spock.questions["TopicB"]["output"]["response"] == "No"
    assert spock.questions["TopicB"]["output"]["sentence"] == "None"



def test_get_topics():
    mock_responses = [
        "Topic1/Topic2"           
    ]
    fake_llm = FakeChatModel(mock_responses)

    spock = Spock(paper="fake-paper.pdf", model="llama3.3", embed_model=True)
    spock.llm = fake_llm
    spock.paper_summary = "This should be a scientific summary."

    spock.get_topics()

    assert spock.paper_summary == "This should be a scientific summary."
    assert spock.topics == "Topic1/Topic2"

def test_format_output():
    spock = Spock(paper="fake-paper.pdf",embed_model=True)
    spock.paper_summary = "This PDF is about advanced screening algorithms."
    spock.topics = "screening algorithms/automation"

    spock.questions = {
        "screening algorithms": {
            "question": "Does it mention screening algorithms?",
            "output": {"response": "Yes", "sentence": "We found it in section 2"}
        },
        "automation": {
            "question": "Any automation techniques used?",
            "output": {"response": "No", "sentence": "None"}
        }
    }

    output = spock.format_output()

    assert "📄 Summary of the Publication" in output
    assert "This PDF is about advanced screening algorithms." in output
    assert "📝 Topics Covered in the Publication" in output
    assert "screening algorithms/automation" in output

    assert "❓ Question: screening algorithms" in output
    assert "💡 Answer: Yes" in output
    assert "🔎 Supporting Sentence: We found it in section 2" in output

    assert "❓ Question: automation" in output
    assert "💡 Answer: No" in output
    assert "🔎 Supporting Sentence: None" in output
