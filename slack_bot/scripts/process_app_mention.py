# TTILE OR DOI
import argparse
import os
from slack_sdk import WebClient
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import getpass
from dotenv import load_dotenv
from texts import COMMANDS

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
