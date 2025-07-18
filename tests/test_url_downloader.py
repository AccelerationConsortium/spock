from spock_literature.core.Spock_Downloader import URLDownloader
import pytest
import os
import re
import requests
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from io import BytesIO


@pytest.fixture
def temp_download_path(tmp_path):
    """A pytest fixture that creates a temporary directory for downloading files."""
    return tmp_path


@pytest.fixture
def url_downloader(temp_download_path):
    """A pytest fixture that returns a URLDownloader instance with a default URL."""
    valid_url = "https://arxiv.org/pdf/1234.5678.pdf"
    return URLDownloader(valid_url, temp_download_path)


@pytest.mark.parametrize("url, expected", [
    ("https://arxiv.org/pdf/1234.5678.pdf", True),
    ("https://biorxiv.org/content/10.1101/2020.01.01.123456v2", True),
    ("https://chemrxiv.org/engage/chemrxiv/article-details/60c74d174c89196d04ad2f0b", True),
    ("http://example.com/some-article", True),
    ("ftp://arxiv.org/pdf/1234.5678.pdf", False),
    ("invalidurl", False),
    ("https://", False),
])
def test_validator(url, expected):
    """Test the static method validator for correct URL validation results."""
    assert URLDownloader.validator(url) == expected


@patch("requests.get")
def test_preprint_download_pdf(mock_get, url_downloader):
    """
    Test __preprint_download when the content is directly a PDF.
    The content-type is set to 'application/pdf' and the status code is 200.
    """
    # Mock the response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {'Content-Type': 'application/pdf'}
    mock_response.content = b"PDF content"
    mock_get.return_value = mock_response
    
    # Call the URLDownloader
    downloaded_file = url_downloader()
    
    # Assert the file was written correctly
    assert downloaded_file.exists()
    assert downloaded_file.suffix == ".pdf"
    with open(downloaded_file, 'rb') as f:
        assert f.read() == b"PDF content"


@patch("requests.get")
def test_preprint_download_html_then_pdf(mock_get, url_downloader):
    """
    Test __preprint_download when the content is HTML first and then we find a PDF link in that HTML.
    """
    # Mock the initial HTML response
    mock_html_response = MagicMock()
    mock_html_response.status_code = 200
    mock_html_response.headers = {'Content-Type': 'text/html'}
    mock_html_response.text = """
        <html>
            <head><title>Test</title></head>
            <body>
                <a href="https://arxiv.org/pdf/1234.5678.pdf">Download PDF</a>
            </body>
        </html>
    """
    
    # Mock the subsequent PDF response
    mock_pdf_response = MagicMock()
    mock_pdf_response.status_code = 200
    mock_pdf_response.content = b"Mock PDF Content"

    # The side_effect allows the first call to return HTML, second to return PDF
    mock_get.side_effect = [mock_html_response, mock_pdf_response]

    downloaded_file = url_downloader()

    assert downloaded_file.exists()
    assert downloaded_file.suffix == ".pdf"
    with open(downloaded_file, 'rb') as f:
        assert f.read() == b"Mock PDF Content"


@patch("requests.get")
def test_preprint_download_html_no_pdf_link(mock_get, url_downloader, caplog):
    """
    Test __preprint_download when the content is HTML and NO PDF link is found.
    This should raise a ValueError indicating no PDF link was found.
    """
    # Mock the HTML response with no PDF link
    mock_html_response = MagicMock()
    mock_html_response.status_code = 200
    mock_html_response.headers = {'Content-Type': 'text/html'}
    mock_html_response.text = """
        <html>
            <head><title>Test</title></head>
            <body>
                <p>No PDF link here!</p>
            </body>
        </html>
    """
    mock_get.return_value = mock_html_response
    downloaded = url_downloader()
    assert downloaded is None
    assert any("Couldn't find PDF link" in record.message for record in caplog.records)


@patch("requests.get")
def test_preprint_download_connection_error(mock_get, url_downloader):
    """
    Test that a ConnectionError is raised if the initial request fails 
    (e.g., status_code != 200).
    """
    mock_bad_response = MagicMock()
    mock_bad_response.status_code = 404
    mock_get.return_value = mock_bad_response

    with pytest.raises(ConnectionError) as e:
        url_downloader()
    assert "Failed to download" in str(e.value)


@patch("requests.get")
def test_journals_download_full_paper(mock_get, temp_download_path):
    """
    Test __journals_download scenario where the HTML is recognized as a 
    'complete scientific paper' by the LLM. We mock the LLM decision to True.
    """
    url = "https://example.com/journal-article"
    # Construct the downloader
    dl = URLDownloader(url, temp_download_path)

    # Mock the HTML response
    mock_html_response = MagicMock()
    mock_html_response.status_code = 200
    mock_html_response.headers = {'Content-Type': 'text/html'}
    mock_html_response.text = """
        <html>
            <head><title>Full Journal Article</title></head>
            <body>
                <h1>Introduction</h1>
                <p>Some relevant science content here...</p>
                <h1>Methods</h1>
                <p>Detailed methods...</p>
                <h1>Results</h1>
                <p>Some results...</p>
                <h1>Discussion</h1>
                <p>Discussion of results...</p>
                <h1>References</h1>
                <p>Reference list...</p>
            </body>
        </html>
    """
    mock_get.return_value = mock_html_response
    with patch.object(dl, 'llm_document_decider', return_value=True):
        result = dl()
        assert not isinstance(result, Path)
        assert hasattr(result, 'page_content')
        assert "Some relevant science content here..." in result.page_content


@patch("requests.get")
def test_journals_download_pdf(mock_get, temp_download_path):
    url = "https://example.com/journal-article-pdf"
    dl = URLDownloader(url, temp_download_path)
    mock_html_response = MagicMock()
    mock_html_response.status_code = 200
    mock_html_response.headers = {'Content-Type': 'text/html'}
    mock_html_response.text = """
        <html>
            <head><title>Journal with PDF Link</title></head>
            <body>
                <p>Some partial content but not full scientific paper.</p>
                <a href="https://example.com/articles/complete.pdf">Download PDF</a>
            </body>
        </html>
    """
    mock_pdf_response = MagicMock()
    mock_pdf_response.status_code = 200
    mock_pdf_response.content = b"Journal PDF content"
    mock_get.side_effect = [mock_html_response, mock_pdf_response]

    with patch.object(dl, 'llm_document_decider', return_value=False):
        result = dl()
        assert (result, Path)
        assert result.exists()
        assert result.suffix == ".pdf"
        with open(result, 'rb') as f:
            assert f.read() == b"Journal PDF content"


@patch("requests.get")
def test_journals_download_no_pdf_link(mock_get, temp_download_path, caplog):
    
    url = "https://example.com/journal-article-no-pdf"
    dl = URLDownloader(url, temp_download_path)
    mock_html_response = MagicMock()
    mock_html_response.status_code = 200
    mock_html_response.headers = {'Content-Type': 'text/html'}
    mock_html_response.text = """
        <html>
            <head><title>No PDF Here</title></head>
            <body>
                <p>Missing PDF link</p>
            </body>
        </html>
    """

    mock_get.return_value = mock_html_response

    with patch.object(dl, 'llm_document_decider', return_value=False):
        result = dl()
        assert result is None
        assert any("Couldn't find PDF link" in record.message for record in caplog.records)


def test_invalid_url_raises_value_error(temp_download_path):
    invalid_url = "not-a-valid-url"
    with pytest.raises(ValueError) as e:
        URLDownloader(invalid_url, temp_download_path)
    assert "Invalid URL" in str(e.value)
