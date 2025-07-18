# Main idea workflow 


document -> md -> document class (parsing could be done through semantic search or something + regex) + cross encoder or something -> Embedding of document -> (give each section to the llm or do rag over it (faiss metadata honestly could be useful)) -> parent page ranking -> reranking thorugh encoder -> to see

- graph rag for methods (allowing spock to do more complex queries)
- text clustering to get sections 
- fine tuning embedding models for dense retrieval
- reranking stuff
- show graph of precedures in thing


Priorities:
1. Document class or whatever (with text clustering and pydantic) -> Could fine tune embedding model here - spacy or something - to see on book how they do it
2. Parent page retrieval (embed doc + summary (or abstract) + hypothetical questions it could answer + chunks) 
3. LLM reranking
4. Query transformation (query routing to consider)
5. Cleaning up the md files (generating json files for the tables + using a better vlm for it)
6. Graph rag for methods
8. Fine tuning embedding models
9. Open source stuff with tensort-rt 
10. Spock as a python library + cleaning up the code
11. Documentation and examples for users
12. Fine-tuning model on the arxiv dataset (optional)
13. Spock with all the AC papers
14. Docker and devops stuff 
15. Return pydantic object for the urldownlaodre
16. Knowledge graph ?????
17. Add ContextualCompressionRetriever ??? Could be useful for the 10 questions stuff 