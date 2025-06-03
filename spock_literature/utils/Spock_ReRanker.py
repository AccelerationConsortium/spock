from typing import List, Optional, Union
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings,OpenAI
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors.rankllm_rerank import RankLLMRerank
from langchain_core.retrievers import BaseRetriever


class Spock_ReRanker:
    def __init__(self, query: str, retriever: BaseRetriever, n: int = 3, **kwargs):
        """
        Initialize the Spock_ReRanker with a query and a retriever.
        """
        self.query = query
        self.retriever = retriever
        self.n = n
        self.kwargs = kwargs
        
        
    def rerank_llm(self, query, model:Union[str, ChatOpenAI], retriever, embedding=OpenAIEmbeddings, n:int=3) -> ContextualCompressionRetriever: # Only supports OpenAI
        """
        Rerank two sentences using a language model.
        """
        
        """
        if model.base_url == "https://localhost/v1": # Check base url
            raise NotImplementedError("Local LLMs are not supported for reranking. GPT from openAi.")
        """
        if isinstance(model, ChatOpenAI):
            raise NotImplementedError("ChatOpenAI is not supported for reranking. Use a string model name instead.")
        compressor = RankLLMRerank(client=None, top_n=n, model="gpt", gpt_model=model)  # Maybe update client here to match Tensortt
        return ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )
        
    def rerank_cross_encoder(self, question, retriever, model="BAAI/bge-reranker-base"):
        """
        Rerank two sentences using a cross-encoder model.
        
        Args:
            sentence_1 (str): The first sentence.
            sentence_2 (str): The second sentence.
            model: The cross-encoder model to use for reranking.
        
        Returns:
            str: The sentence that is ranked higher by the model.
        """
        model = HuggingFaceCrossEncoder(model_name=model)
        compressor = CrossEncoderReranker(model=model, top_n=3)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )
        raise NotImplementedError("Cross-encoder reranking is not implemented yet.")
        compressed_docs = compression_retriever.invoke(question)
        pretty_print_docs(compressed_docs)
