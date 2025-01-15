import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from spock_literature.utils import Author
from typing import List


def is_list_of_dicts_of_strs(data) -> bool:
    return (
        isinstance(data, list) and
        all(isinstance(item, dict) and
            all(isinstance(k, str) and isinstance(v, str) for k, v in item.items())
            for item in data)
    )



def test_author_call():
    author = Author(
        "Mehrad Ansari"
        )

    for item in author.get_last_publication(1):
        assert "title" in item, "'title' key missing in result dictionary"
        assert "abstract" in item, "'abstract' key missing in result dictionary"
        assert "author" in item, "'author' key missing in result dictionary"
        assert "year" in item, "'year' key missing in result dictionary"
        assert "url" in item, "'url' key missing in result dictionary"

