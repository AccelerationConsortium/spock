from spock import Spock


class Spock_litellm(Spock):
    def __init__(self, model = "llama3.3", paper = None, custom_questions = None, publication_doi = None, publication_title = None, publication_url = None, papers_download_path = ..., temperature = 0.2, embed_model=None, folder_path=None, settings = ...):
        super().__init__(model, paper, custom_questions, publication_doi, publication_title, publication_url, papers_download_path, temperature, embed_model, folder_path, settings)
        
        
    def summarize(self): # to edit using litellm
        return super().summarize()