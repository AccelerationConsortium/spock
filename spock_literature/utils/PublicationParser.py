from langchain_core.output_parsers import BaseOutputParser
import re
from spock_literature.utils.Publication import Publication


class PublicationOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        pass
    
    @property
    def _type(self) -> str:
        return "Publication Output Parser"
