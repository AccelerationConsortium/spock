"""The common module contains common functions and classes used by the other modules.
"""

try:
    from .author import Author
    from .publication import Publication
    from .bot import Bot
except:
    from author import Author
    from publication import Publication
    from bot import Bot
    

def setup_json(author):
    import concurrent.futures

    try:
        author = author[:-1]
        author_filled = Author(author)
        print('Author created successfully for ' + author)
        author_filled.setup_author('json/output.json')
        print(f"Topics for {author} have been updated")
    except Exception as e:
        print(f"Couldn't find the google scholar profile for {author}: {e}")

def setup() -> None:
    import concurrent.futures
    with open("data/authors.txt","r") as file:
        authors = file.readlines()
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:  # Adjust max_workers as needed
        executor.map(setup_json, authors)




def download_pdfs(publication: Publication, local_file_path: str):
    import requests

    try:
        # Send a GET request to the URL
        response = requests.get(publication.pdf)

        # Check if the request was successful
        if response.status_code == 200:
            # Write the content to a local file
            with open(local_file_path, 'wb') as file:
                file.write(response.content)
            print(f"PDF successfully downloaded and saved to {local_file_path}")
        else:
            print(f"Failed to download the PDF. HTTP Status Code: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while downloading the PDF: {e}")



# à revoir
def rag(file):
    from langchain_community.llms import Ollama
    from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PDFPlumberLoader
    from langchain_community.vectorstores import Chroma
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains import create_retrieval_chain
    from langchain.prompts import PromptTemplate
    
    # Set up
    llm = Ollama(model = "llama3")
    folder_path = "db"
    prompt = PromptTemplate.from_template(    """ 
        <s>[INST] You are a technical assistant good at searching docuemnts. If you do not have an answer from the provided information say so. [/INST] </s>
        [INST] {input}
            Context: {context}
            Answer:
        [/INST]
    """)
    embedding = FastEmbedEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(    chunk_size=1024,
    chunk_overlap=80,
    length_function=len,
    is_separator_regex=False,
)
    def pdf_loader(file):
        loader = PDFPlumberLoader(file)
        docs = loader.load_and_split()

        # Getting the chunks

        chunks = text_splitter.split_documents(docs)
        vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=folder_path
    )

        vector_store.persist()
        vectore_store = Chroma(persist_directory=folder_path, embedding_function=embedding)

        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 20,
                "score_threshold": 0.1,
            },
        )

        document_chain = create_stuff_documents_chain(llm, prompt)
        chain = create_retrieval_chain(retriever, document_chain)
        query = "Who is the Author"
        result = chain.invoke({"input": query})

        sources = []
        for doc in result["context"]:
            sources.append(
                {"source": doc.metadata["source"], "page_content": doc.page_content}
            )

        response_answer = {"answer": result["answer"]}
        print(response_answer)

    

    
