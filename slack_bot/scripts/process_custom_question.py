import argparse
import os
from slack_sdk import WebClient
from spock_literature import Spock
from langchain_community.callbacks import get_openai_callback
import time
import json


USERS_FILE = "/home/m/mehrad/brikiyou/scratch/spock/slack_bot/users.json"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--paper', required=True)
    parser.add_argument('--questions', default="")
    parser.add_argument('--user_id', required=True)
    parser.add_argument('--channel_id', required=True)
    args = parser.parse_args()

    model = args.model
    paper_path = args.paper
    questions_str = args.questions
    user_id = args.user_id
    channel_id = args.channel_id
