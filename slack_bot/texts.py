PAPERS_PATH = "/home/m/mehrad/brikiyou/scratch/spock/slack_bot/papers/"
USER_JSON_PATH = "/home/m/mehrad/brikiyou/scratch/spock/slack_bot/users.json"
SCRIPTS_PATH = "/home/m/mehrad/brikiyou/scratch/spock/slack_bot/scripts/"
GENERATED_SCRIPTS_PATH = "/home/m/mehrad/brikiyou/scratch/spock/slack_bot/scripts/generated_job_scripts/"
COMMANDS = """

- 📚 get_author_publication: Get the latest publications of a specific author. By default, it fetches the most recent one. To specify how many publications you'd like, provide the author's name followed by a comma and the number of publications.
  Example: /get_author_publication Jane Doe, 3 (This will retrieve the 3 latest publications by Jane Doe.)

- 📄 process_pdf: Share and process a PDF file from your local computer. You can also include custom questions about the PDF by writing them next to the command, separated by a `||`.
  Example: /process_pdf Does the paper discuss the impact of AI on society? || What are the key findings of the paper?

- 📖 process_publication: Process a publication by providing its title, DOI, or URL. You can also include custom questions about the publication by writing them next to the command, separated by a `||`.  
    Example: /process_publication Title of the publication OR DOI of the publication OR URL of the publication Does the paper discuss the impact of AI on society? || What are the key findings of the paper?

- 🤖 choose_llm: Choose the language model you'd like to use for future responses. You can choose between Llama3.1, Claude 3.5 Sonnet, and GPT-4. To do so, type /choose_llm followed by the model name.
  Example: /choose_llm llama3.3 OR /choose_llm gpt-4o
- 🎙️ generate_podcast: Generate a podcast from a text input.
"""
