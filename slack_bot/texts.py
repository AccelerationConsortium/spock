PAPERS_PATH = "/home/m/mehrad/brikiyou/scratch/spock/slack_bot/papers/"
USER_JSON_PATH = "/home/m/mehrad/brikiyou/scratch/spock/slack_bot/users.json"
SCRIPTS_PATH = "/home/m/mehrad/brikiyou/scratch/spock/slack_bot/scripts/"
GENERATED_SCRIPTS_PATH = "/home/m/mehrad/brikiyou/scratch/spock/slack_bot/scripts/generated_job_scripts/"
COMMANDS = """
- 🆘 **/help**  
  Displays helpful information about the bot and an overview of all available commands.
  Example: /help

- 📚 **/get_author_publication**  
  Get the latest publications of a specific author. By default, it fetches the most recent one. To specify how many publications you'd like, provide the author's name followed by a comma and the number of publications.  
  Example: /get_author_publication Jane Doe, 3  
  (This will retrieve the 3 latest publications by Jane Doe.)

- 📄 **/process_pdf**  
  Share and process a PDF file from your local computer. You can also include custom questions about the PDF by writing them next to the command, separated by `||`.  
  Example: /process_pdf Does the paper discuss the impact of AI on society? || What are the key findings of the paper?

- 📖 **/process_publication**  
  Process a publication by providing its title, DOI, or URL. You can also include custom questions about the publication by writing them next to the command, separated by `||`.  
  Example: /process_publication Title of the publication OR DOI of the publication OR URL of the publication || Does the paper discuss the impact of AI on society?

- 🤖 **/choose_llm**  
  Choose the language model you'd like to use for future responses. Currently, you can choose between **llama3.3** or **gpt-4o**.  
  Example: /choose_llm llama3.3 or /choose_llm gpt-4o

- 🎙️ **/generate_podcast**  
  Generate a podcast (audio file) from a PDF. After typing `/generate_podcast`, upload the PDF when prompted. The bot will then convert the document into an audio summary.

- ⚙️ **/get_user_settings**  
  View your current user settings, including which language model you’re using.  
  Example: /get_user_settings

- 🔀 **/toggle_binary_responses**  
  Toggle binary (yes/no) responses on or off. This setting affects how succinct or expanded the bot's responses will be.  
  Example: /toggle_binary_responses

- 📑 **/toggle_summary**  
  Toggle summary generation on or off. When turned on, the bot will include a summary with each response (where applicable).  
  Example: /toggle_summary
"""

