
from typing import Generator, Optional, Dict, Any
from pydantic import BaseModel, Field, HttpUrl, PositiveInt
from pydantic.dataclasses import dataclass
import json
from scholarly import scholarly


@dataclass(frozen=True, eq=False)
class GScholarPublicationObject:
    title: str = Field(..., description="The title of the publication.")
    abstract: str = Field(..., description="The abstract of the publication.")
    author: str = Field(..., description="The author(s) of the publication.")
    year: PositiveInt = Field(
        ...,
        description="The year of publication. Must be a positive integer.")
    url: Optional[HttpUrl] = Field(default=None, description="The URL of the publication.")
    citation: str = Field(
        ...,
    )
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GScholarPublicationObject":
        """
        Construct from a raw dict that may contain 'publication_filled'.
        If get_topic=True, you could extend this to call an LLM to assign 'topic'.
        """
        raw = data.get("publication_filled", {})
        bib = raw.get("bib", {})
        year = bib.get("pub_year")
        obj = cls(
            title=bib.get("title", ""),
            abstract=bib.get("abstract", ""),
            author=bib.get("author", ""),
            year=int(year) if year is not None else 0,
            url=raw.get("pub_url"),
            citation=bib.get("citation", ""),
        )
        return obj

    @property
    def summary(self) -> str:
        """A one-line summary combining title and year."""
        return f"{self.title} ({self.year})"

    def to_json(self) -> str:
        """Serialize this object to a JSON string."""
        return json.dumps(self.__dict__)

    @staticmethod
    def from_json(json_str: str) -> "GScholarPublicationObject":
        """
        Deserialize a JSON string to a GScholarPublication object.
        """
        data = json.loads(json_str)
        return GScholarPublicationObject(
            title=data.get("title", ""),
            abstract=data.get("abstract", ""),
            author=data.get("author", ""),
            year=data.get("year", 0),
            url=data.get("url"),
            citation=data.get("citation", ""),
        )

@dataclass(frozen=True, eq=False)
class GScholar_Author(BaseModel):
    author_name: str = Field(
        ...,
        description="The name of the author to search for in Google Scholar.",
    )
    
    def get_last_publications(self, count) -> dict:
        """
        Get the last publication of the author.

        Returns:
            dict: A dict containing information about the last publication.
        """
        try:
            publications_filled = []
            search_query = scholarly.search_author(self.author_name)
            first_author_result = next(search_query)
            author = scholarly.fill(first_author_result)
            publications = sorted(author['publications'], 
                                    key=lambda x: int(x['bib']['pub_year'])
                                    if 'pub_year' in x['bib'] else 0, 
                                    reverse=True)[0:count]
            for publication in publications:
                publications_filled.append(scholarly.fill(publication))
            return publications_filled
        except Exception as e:
            print(f"An error occurred, couldnt get the latest publications: {e}")

    def __call__(self, count: int = 1) -> Generator[GScholarPublicationObject, None, None]:
        """
        Get the n last publications of the author.
        """       
        author_publications = self.get_last_publications(count)
        for publication in author_publications:
            yield GScholarPublicationObject(publication)
        