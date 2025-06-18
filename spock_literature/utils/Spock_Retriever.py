from typing import List, Optional, Any, Dict, Union
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ParentDocumentRetriever, MultiVectorRetriever, EnsembleRetriever
from langchain.storage import InMemoryByteStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import faiss
import os
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore import InMemoryDocstore
from langchain.storage import InMemoryByteStore
from pydantic import BaseModel, Field


            """
            # Batch chain over documents to generate hypothetical questions
            hypothetical_questions = chain.batch(docs, {"max_concurrency": 5})


            # The vectorstore to use to index the child chunks
            vectorstore = Chroma(
                collection_name="hypo-questions", embedding_function=OpenAIEmbeddings()
            )
            # The storage layer for the parent documents
            store = InMemoryByteStore()
            id_key = "doc_id"
            # The retriever (empty to start)
            retriever = MultiVectorRetriever(
                vectorstore=vectorstore,
                byte_store=store,
                id_key=id_key,
            )
            doc_ids = [str(uuid.uuid4()) for _ in docs]


            # Generate Document objects from hypothetical questions
            question_docs = []
            for i, question_list in enumerate(hypothetical_questions):
                question_docs.extend(
                    [Document(page_content=s, metadata={id_key: doc_ids[i]}) for s in question_list]
                )


            retriever.vectorstore.add_documents(question_docs)
            retriever.docstore.mset(list(zip(doc_ids, docs)))
            sub_docs = retriever.vectorstore.similarity_search("justice breyer")
            retrieved_docs = retriever.invoke("justice breyer")
            len(retrieved_docs[0].page_content)
            """

class HypotheticalQuestions(BaseModel):
    """Schema for hypothetical questions generation."""
    questions: List[str] = Field(description="List of hypothetical questions")


class Spock_Retriever(BaseRetriever):
    """
    A comprehensive retriever implementation for the Spock system.
    
    Supports multiple retrieval strategies including:
    - Standard chunk retrieval
    - Parent-child document retrieval  
    - Abstract-based retrieval
    - Hypothetical question retrieval
    - Ensemble retrieval combining multiple strategies
    """

    def __init__(self, 
                 retrievers:List[BaseRetriever], 
                 **kwargs):
        super().__init__(**kwargs) 
        self.retrievers = retrievers
        
        
        
    @classmethod
    def from_hypo_questions(cls, 
                             embed_model, 
                             child_splitter=None, 
                             parent_splitter=None, 
                             mode="child", 
                             k=10, 
                             search_type="mmr", 
                             **kwargs) -> "Spock_Retriever":
        """
        Create a Spock_Retriever for hypothetical questions.
        
        Args:
            embed_model: Embedding model for vectorization.
            child_splitter: Text splitter for child documents.
            parent_splitter: Text splitter for parent documents.
            mode: Retrieval mode ('parent', 'child', 'hybrid').
            k: Number of documents to retrieve.
            search_type: Type of search to perform.
            
        Returns:
            An instance of Spock_Retriever.
        """
        return Spock_Retriever(
            embed_model=embed_model,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            mode=mode,
            k=k,
            search_type=search_type,
            docstore=None,
            store=None,
            vectorstore=None
        )
        
        
        
    @classmethod
    def from_abstract_retriever(cls,
                                    embed_model, 
                                    child_splitter=None, 
                                    parent_splitter=None, 
                                    mode="child", 
                                    k=10, 
                                    search_type="mmr", 
                                    **kwargs) -> "Spock_Retriever":
            """
            Create a Spock_Retriever for abstract-based retrieval.
            
            Args:
                embed_model: Embedding model for vectorization.
                child_splitter: Text splitter for child documents.
                parent_splitter: Text splitter for parent documents.
                mode: Retrieval mode ('parent', 'child', 'hybrid').
                k: Number of documents to retrieve.
                search_type: Type of search to perform.
                
            Returns:
                An instance of Spock_Retriever.
            """
            return Spock_Retriever(
                embed_model=embed_model,
                child_splitter=child_splitter,
                parent_splitter=parent_splitter,
                mode=mode,
                k=k,
                search_type=search_type,
                docstore=None,
                store=None,
                vectorstore=None
            )
            
            
        self.abstract_retriever = MultiVectorRetriever( # Maybe to add summary to vectorstore after being computed
            vectorstore=vectorstore,
            byte_store=self.store,
            id_key=id_key,
            )
        
        self.hypothetical_question_retriever = MultiVectorRetriever(
        )


    
    @classmethod
    def from_retrievers(cls, 
                        retrievers: List[BaseRetriever], 
                        embed_model, 
                        child_splitter=None, 
                        parent_splitter=None, 
                        mode="child", 
                        k=10, 
                        search_type="mmr", 
                        **kwargs) -> "Spock_Retriever":
        """
        Create a Spock_Retriever from multiple retrievers.
        
        Args:
            retrievers: List of BaseRetriever instances to combine.
            embed_model: Embedding model for vectorization.
            child_splitter: Text splitter for child documents.
            parent_splitter: Text splitter for parent documents.
            mode: Retrieval mode ('parent', 'child', 'hybrid').
            k: Number of documents to retrieve.
            search_type: Type of search to perform.
            
        Returns:
            An instance of Spock_Retriever.
        """
        return Spock_Retriever(
            embed_model=embed_model,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            mode=mode,
            k=k,
            search_type=search_type,
            docstore=None,
            store=None,
            vectorstore=None
        )
        
        
        


    @staticmethod
    def create_hypothetical_questions_retriever() -> MultiVectorRetriever:
        
        """
                self.abstract_retriever = MultiVectorRetriever( # Maybe to add summary to vectorstore after being computed
            vectorstore=vectorstore,
            byte_store=self.store,
            id_key=id_key,
            )
        
        self.hypothetical_question_retriever = MultiVectorRetriever(
        )
        
    """
    
    @staticmethod
    def create_retriever_from_abstract():
        
        """
        self.doc_retriever = ParentDocumentRetriever(
                        vectorstore=vectorstore,
                        docstore=self.store,
                        child_splitter=child_splitter,
                    )
        self.abstract_retriever = MultiVectorRetriever( # Maybe to add summary to vectorstore after being computed
            vectorstore=vectorstore,
            byte_store=self.store,
            id_key=id_key,
            )
        
        """

    def _initialize_stores(self):
        """Initialize the vector store and document stores."""
        raise NotImplementedError("This method should be implemented in subclasses.")
        
        if self.docstore is None:
            self.docstore = InMemoryDocstore({})
        
        if self.store is None:
            self.store = InMemoryByteStore()
        
        if self.vectorstore is None:
            embedding_dim = len(self.embed_model.embed_query("hello world"))
            
            self.vectorstore = FAISS(
                embedding_function=self.embed_model,
                index=faiss.IndexFlatL2(embedding_dim),
                docstore=self.docstore,
                index_to_docstore_id={}
            )
        
        self.chunk_retriever = self.vectorstore.as_retriever(
            search_type=self.search_type,
            search_kwargs=self.search_kwargs,
        )    def add_to_vectorstore(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store with parent-child relationship support.
        
        Args:
            documents: A list of Document objects to be added to the vector store.
        """
        if not documents:
            return
            
        child_docs = []
        
        for doc in documents:
            parent_id = doc.metadata.get(self.id_key) or f"parent_{len(self.docstore)}"
            
            parent_doc = doc.model_copy()
            parent_doc.metadata[self.id_key] = parent_id
            self.docstore[parent_id] = parent_doc
            
            child_chunks = self.child_splitter.split_documents([doc])
            
            for i, child_doc in enumerate(child_chunks):
                child_id = f"{parent_id}_child_{i}"
                child_doc.metadata[self.id_key] = child_id
                child_doc.metadata["parent_id"] = parent_id
                child_docs.append(child_doc)
        
        # Add child documents to vector store
        if child_docs:
            if len(self.vectorstore.index_to_docstore_id) == 0:
                # First batch of documents
                self.vectorstore = FAISS.from_documents(child_docs, self.embeddings)
            else:
                # Add to existing vectorstore
                self.vectorstore.add_documents(child_docs)
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        """
    
    async def _aget_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Async version of document retrieval.
        
        Args:
            query: The search query string
            run_manager: Async callback manager for the retrieval run
            
        Returns:
            List of relevant documents based on the configured mode
        """
        raise NotImplementedError("Async retrieval is not implemented yet.")
        return self._get_relevant_documents(query, run_manager=run_manager.get_sync())
    
    def get_vectorstore_stats(self) -> dict[str, Any]:
        """Get statistics about the vectorstore and docstore."""
        return {
            "vectorstore_size": len(self.vectorstore.index_to_docstore_id) if self.vectorstore else 0,
            "docstore_size": len(self.docstore),
            "mode": self.mode,
            "k": self.k
        }
    
    def clear_stores(self) -> None:
        """Clear both vectorstore and docstore."""
        self.docstore.clear()
        if self.vectorstore:
            # Reinitialize empty vectorstore
            self.vectorstore = FAISS.from_texts([""], self.embeddings)
            self.vectorstore.delete([0])


"""
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# Initialize components
embeddings = OpenAIEmbeddings()
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

# Create retriever
retriever = Spock_Retriever(
    embeddings=embeddings,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    mode="parent",  # or "child" or "hybrid"
    k=5
)

# Add documents
documents = [Document(page_content="Your document content...", metadata={"source": "doc1"})]
retriever.add_to_vectorstore(documents)

# Retrieve documents
results = retriever.invoke("your query here")
"""
