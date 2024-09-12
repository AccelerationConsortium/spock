import os
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA


class LLM:
    def __init__(self, llm_model:str="llama3", embedding_model:str="mxbai-embed-large") -> None :
        self.llm = Ollama(model=llm_model, temperature=0.2)
        self.embedding = OllamaEmbeddings(model=embedding_model)
        self.folder_path = None
        
    def set_folder_path(self, folder_path:str) -> None:
        self.folder_path = folder_path
        
        
    def pdf_to_md(self, pdf_path:str) -> str:
        nougat_cmd = f"nougat --markdown pdf '{pdf_path}' --out 'papers/mmd'"
        os.system(nougat_cmd)
        
        
        
    def split_markdown(self, md_path:str):
        markdown_document = open(md_path, "r").read()
        
        # Maybe add header/update
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
        ]

        # MD splits
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, strip_headers=False
        )
        md_header_splits = markdown_splitter.split_text(markdown_document)

        # Char-level splits
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        chunk_size = 1000
        chunk_overlap = 50
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        # Split
        splits = text_splitter.split_documents(md_header_splits)
        return splits
        
    def embedding_chunk(self, chunks):
        if self.folder_path is None:
            raise ValueError("Please set the folder path using the set_folder_path method.")
        self.vectorstore = Chroma.from_documents(documents=chunks, embedding=self.embedding, persist_directory=self.folder_path)
        
    def query_rag(self, question:str, **kwags) -> None:
        try:
            docs = self.vectorstore.similarity_search(question)
            from langchain.chains import RetrievalQA
            qachain=RetrievalQA.from_chain_type(self.llm, retriever=self.vectorstore.as_retriever(), verbose=True)
            res = qachain.invoke({"query": question})
            print(res['result'])
            return res['result']


        except Exception as e:
            print(e)


            
