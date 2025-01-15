# TTILE OR DOI
import argparse
import os
from slack_sdk import WebClient
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import getpass
import os
from dotenv import load_dotenv

COMMANDS = """- 📜 get_authors_list: Retrieve a list of all authors.
  Example: /get_authors_list

- 📚 get_author_publication: Get the latest publications of a specific author. By default, it fetches the most recent one. To specify how many publications you'd like, provide the author's name followed by a comma and the number of publications.
  Example: /get_author_publication Jane Doe, 3 (This will retrieve the 3 latest publications by Jane Doe.)

- 📄 process_pdf: Share and process a PDF file from your local computer. You can also include custom questions about the PDF by writing them next to the command, separated by a `||`.
  Example: /process_pdf Does the paper discuss the impact of AI on society? / What are the key findings of the paper?

- 📖 process_publication: Process a publication by providing its title, DOI, or URL. You can also include custom questions about the publication by writing them next to the command, separated by a `||`.  
    Example: /process_publication Title of the publication / DOI of the publication / URL of the publication || Does the paper discuss the impact of AI on society? || What are the key findings of the paper?

- 🤖 choose_llm: Choose the language model you'd like to use for future responses. You can choose between Llama3.1, Claude 3.5 Sonnet, and GPT-4. To do so, type /choose_llm followed by the model name.

- 🎙️ generate_podcast: Generate a podcast from a text input. You would do /generate_podcast, then input your PDF file.
"""
load_dotenv()
def get_api_key(env_var, prompt):
    
    if not os.getenv(env_var):
        os.environ[env_var] = getpass.getpass(prompt)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--question', required=True)
    parser.add_argument('--user_id', required=True)
    parser.add_argument('--channel_id', required=True)
    args = parser.parse_args()

    question = args.question
    user_id = args.user_id
    channel_id = args.channel_id
    
    # Send the response back to the user via Slack
    BOT_TOKEN = os.getenv('BOT_TOKEN')
    client = WebClient(token=BOT_TOKEN)


    # Prepare custom questions
    if not question:
        client.chat_postMessage(
        channel=channel_id,
        text=f"Hi there, <@{user_id}>! Please provide a question",
        )
        return
    
    get_api_key("OPENAI_API_KEY", "Enter your OpenAI API key: ")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


    prompt = PromptTemplate(
        template=f"""You are an assistant of a slack bot. Here is what can the bot do Commands: {COMMANDS}
 and here is someone asking you a question. Please provide a response to the following question: {{question}}""",
        input_variables=["question"]
    )

    chain = prompt | llm
    response = chain.invoke({"question": question}).content
    
    client.chat_postMessage(
        channel=channel_id,
        text=f"Hi there, <@{user_id}>! Here is the response to your question: {response}",
        mrkdwn=True
    )

if __name__ == "__main__":
    main()
