import spock_literature

def test_import():
    try:
        from spock_literature.utils import Publication_scholarly, Author, Helper_LLM
        from spock_literature import Spock
    except ImportError as e:
        assert False, f"Failed to import module: {e}"
        
