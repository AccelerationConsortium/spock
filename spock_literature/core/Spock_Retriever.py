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

# Hypothetical Questions Generation Example
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
        
        
    def _get_relevant_documents(
        self,
        query: str, 
        run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """Get documents relevant to a query.

        Args:
            query: String to find relevant documents for.
            run_manager: The callback handler to use.
        Returns:
            List of relevant documents.
        """
        ensemble_retriever = EnsembleRetriever(retrivers=self.retrievers, weights=weights)

        
        
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

    def add_to_vectorstore(
        self,
        parent_retrieval: bool = False,
        abstract_retrieval: bool = False,
        hypothetical_question_retrieval: bool = False,
        use_semantic_splitting: bool = False,
        search_type: str = "mmr",   
    ) -> Any: # Returns List[Retrievers] to see how it's done on langchain (datatype)
        
        retrievers = []
        self.store = InMemoryByteStore()
        splitter = self.__get_splitter(use_semantic_splitting)
        
        if parent_retrieval:
            parent_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2250, chunk_overlap=150
            )
            self.parent_retriever = ParentDocumentRetriever(
                                    vectorstore=self.vectorstore,
                                    docstore=self.store,
                                    child_splitter=spliter,
                                    parent_splitter=parent_splitter,
                                    #search_type=SearchType.mmr  # to see 
                                )
            retrievers.append(('parent', self.parent_retriever))
            #self.parent_retriever.add_documents(self.paper) # To change 
        if abstract_retrieval:
            self.abstract_retriever = MultiVectorRetriever(
                vectorstore=self.vectorstore,
                byte_store=self.store,
                #id_key="id",
                #search_type=SearchType.mmr if search_type == "mmr" else SearchType.hnsw
            )
            abstract = self.paper.abstract # To update
            # TODO: work on adding the original doc too
            retrievers.append(('abstract', self.abstract_retriever))
            
        if hypothetical_question_retrieval:
            
            # Pass in the abstract
            chain = (
                {"doc": lambda x: x.page_content} 
                # Only asking for 3 hypothetical questions, but this could be adjusted
                | PromptTemplate.from_template(
                    "Generate a list of exactly 3 hypothetical questions that the below document could be used to answer:\n\n{doc}"
                )
                | self.llm.with_structured_output(
                    HypotheticalQuestions
                )
                | (lambda x: x.questions)
            )
            retrievers.append(('hypothetical_questions', chain))
            
        self.available_retrievers = dict(retrievers)
        return retrievers

            
    def __get_available_retrievers(self, weights:Optional[Dict[str, float]]) -> Dict[str, Any]:
        """
        Get the available retrievers based on the current configuration.
        """
        retrievers = {}
        
        # To fix weights        
        if hasattr(self, 'parent_retriever') and self.parent_retriever:
            retrievers['parent'] = {
                'retriever': self.parent_retriever,
                'description': 'Parent-child document retrieval',
                'weight': weights.get('parent', 0.3)
            }
        
        if hasattr(self, 'abstract_retriever') and self.abstract_retriever:
            retrievers['abstract'] = {
                'retriever': self.abstract_retriever,
                'description': 'Abstract-based retrieval',
                'weight': weights.get('abstract', 0.2)
            }
        
        if hasattr(self, 'hypo_retriever') and self.hypo_retriever:
            retrievers['hypothetical'] = {
                'retriever': self.hypo_retriever,
                'description': 'Hypothetical question retrieval',
                'weight': weights.get('hypothetical', 0.2)
            }
        
        # Base retriever 
        retrievers['chunk'] = {
            'retriever': self.chunk_retriever,
            'description': 'Standard chunk retrieval',
            'weight': weights.get('chunk', 0.3)
        }
        
        return retrievers

    async def aadd_to_vectorstore(
        self,
        settings: dict,
        documents: List[Document]):
        pass 
    
    # To see later on how good it is
    def __create_ensemble_retriever(self, custom_weights: Optional[Dict[str, float]] = None):
        """Create ensemble retriever with available retrievers"""
        available = self.__get_available_retrievers()
        
        if len(available) < 2:
            print("Warning: Only one retriever available, ensemble not beneficial")
            return list(available.values())[0]['retriever']
        
        retrievers = []
        weights = []
        
        for name, config in available.items():
            retrievers.append(config['retriever'])
            weight = custom_weights.get(name, config['weight']) if custom_weights else config['weight']
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w/total_weight for w in weights]
        
        print(f"Creating ensemble with {len(retrievers)} retrievers:")
        for i, (name, _) in enumerate(available.items()):
            print(f"  - {name}: {weights[i]:.2f}")
        
        return EnsembleRetriever(retrievers=retrievers, weights=weights)
