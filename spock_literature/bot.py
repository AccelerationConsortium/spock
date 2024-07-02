from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_sdk import WebClient
from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse

'''
try:
    from .author import Author
    from .publication import Publication
except:
    from author import Author
    from publication import Publication
'''
class Bot:
    def __init__(self, slack_bot_token:str, slack_app_token:str, channel_id:str):
        self.slack_bot_token = slack_bot_token
        self.client = WebClient(token=self.slack_bot_token)
        self.channel_id = channel_id
        self.slack_app_token = slack_app_token
        self.socket_mode_client = SocketModeClient(app_token=self.slack_app_token)
        self.register_handler_to_client()
        
    def process_slash_command(self,payload):
        command = payload['command']
        user_id = payload['user_id']
        text = payload['text']
        channel_id = payload['channel_id']

        if command == '/hello':
            response_message = f"Hello <@{user_id}>!"

            try:
                # Post the message
                self.client.chat_postMessage(
                    channel=channel_id,
                    text=response_message
                )
                print("/hello was successfully posted")
            except SlackApiError as e:
                print(f"Error posting message: {e.response['error']}")
                
        elif command == '/setup':
            response_message = f"Hello <@{user_id}>! It's loading Data, it might take some time"
            try:
                # Post the message
                self.client.chat_postMessage(
                    channel=channel_id,
                    text=response_message
                )
                print("/setup was successfully posted")
                setup() # This function is not defined yet
                self.client.chat_postMessage(
                    channel=channel_id,
                    text="Set up is complete"
                )

            except SlackApiError as e:
                print(f"Error posting message: {e.response['error']}")

    def handle_socket_mode_request(self, req: SocketModeRequest):
        if req.type == "slash_commands":
            self.process_slash_command(req.payload)
            self.socket_mode_client.send_socket_mode_response(SocketModeResponse(envelope_id=req.envelope_id))
        
    def register_handler_to_client(self):
        self.socket_mode_client.socket_mode_request_listeners.append(self.handle_socket_mode_request)    

                
        
def process_scholar(scholar,Bot: Bot):
    key = scholar[0]
    value = scholar[1]
    try:

        author = Author(key)
        #print(f'value title= {value["title"]} \n author title = {author.get_last_publication()["bib"]["title"]}')
        if value['title'] != author.get_last_publication()['bib']['title']:
            
            print(f"Updating topics for {author}")
            
            try:
                last_publication = Publication(author.get_last_publication())
            except Exception as e:
                print(f"Couldn't fetch the last publication for {author}: {e}")
                
            
            text_message = f":rolled_up_newspaper::test_tube: {author.author_name} has an update on Google Scholar!\n\
                    ```Title: {last_publication.title}\nCitation: {last_publication.citation}\nYear: {last_publication.year}```"
            try:
                response = Bot.client.chat_postMessage(
                channel=Bot.channel_id, 
                text=text_message)
            except Exception as e:
                print(f"Couldn't send the message to slack: {e}")
            
            # Updating the Json file
            try:
                author.setup_author('json/ouput.json')
            except Exception as e:
                print(f"Couldn't Overwrite the old data for: {author}: {e}")

        
        print(f"Topics for {author} have been updated")
    except Exception as e:
        print(f"Couldn't find the google scholar profile for {author}: {e}")


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader, TextLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.llms import Ollama
import json
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate


class Bot_LLM:
    def __init__(self,model='llama3',embed_model='mxbai-embed-large', folder_path='db2'):
        self.llm = Ollama(model=model)
        self.oembed = OllamaEmbeddings(model=embed_model)
        self.folder_path = folder_path
        self.vectorestore = None

    
    def get_topic_publication_abstract(self, abstract:str, input_file:str):
        with open(input_file, 'r') as file:
            data = json.load(file)
        
        parser = JsonOutputParser()
        
        new_text = """The output should be formatted as a JSON instance that conforms to the JSON schema below.

        As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
        the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

        Here is the output schema:
        ```
        {"topic": {'Machine Learning: [Keyword1, keyword2, keyword3], 'Batteries: [keyword1, keyword2, keyword3]}
        ```
        """


        prompt = PromptTemplate(
            template="Here is a text: {abstract} Please identify the topics from the following list: {liste}. Note: A single text can belong to multiple topics, so please list all relevant topics.  \n{format_instructions}"
        ,
            input_variables=["abstract","liste","topics"],
            partial_variables={"format_instructions": new_text}
        )


        chain = prompt | self.llm | parser
        topics = chain.invoke({"abstract": abstract, "liste": data.keys()})
        print('Topics: ', topics['topic'])
        return topics['topic']

    
    def rag(self, document:str):
        try:
            # The document is a pdf file
            loader = PDFPlumberLoader(document)
            data = loader.load()
            chunk_size = 500
            chunk_overlap = 20

        except:
            data = [TextLoader(text).load() for text in document]
            data = [item for sublist in data for item in sublist]

            
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_splits = text_splitter.split_documents(data)
        self.vectorstore = Chroma.from_documents(documents=all_splits, embedding=self.oembed, persist_directory=self.folder_path)
        
    def query_rag(self, question:str) -> None:
        if self.vectorstore:
            docs = self.vectorstore.similarity_search(question)
            from langchain.chains import RetrievalQA
            qachain=RetrievalQA.from_chain_type(self.llm, retriever=self.vectorstore.as_retriever(), verbose=True)
            res = qachain.invoke({"query": question})
            print(res['result'])


        else:
            raise Exception("No documents loaded")



        
