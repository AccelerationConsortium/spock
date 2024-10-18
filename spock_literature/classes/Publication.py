from abc import ABC, abstractmethod

class Publication(metaclass=ABCMeta):
    @abstractmethod
    def get_publication_url(self) -> str:
        pass
    
    @abstractmethod
    def get_publication_title(self) -> str:
        pass

    @abstractmethod
    def get_publication_abstract(self) -> str:
        pass

    @abstractmethod
    def get_author_name(self) -> str:
        pass

    @abstractmethod
    def get_year(self) -> str:
        pass
    
    @abstractmethod
    def get_citation(self) -> str:
        pass
    
    @abstractmethod
    def get_topic(self,llm):
        pass