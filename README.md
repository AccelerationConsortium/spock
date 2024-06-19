# spock

[![image](https://img.shields.io/pypi/v/spock.svg)](https://pypi.python.org/pypi/spock)
[![image](https://img.shields.io/conda/vn/conda-forge/spock.svg)](https://anaconda.org/conda-forge/spock)

**LLM Literature**

-   Free software: MIT License
-   Documentation: [Spock Documentation](https://youssefbriki1.github.io/spock)
    
## Project Presentation:

An LLM-powered Slack bot for monitoring LLM literature and AC members' publications (e.g., Heisenberg).

### Features:

- Notifies whenever an AC member publishes a paper.
- Provides the topics of AC members' papers.
- Retrieves information from scientific papers.
- Allows interaction from Slack chat.
- Monitors and analyzes various types of publications and outputs by Acceleration Consortium Members, including:
  - Peer-reviewed publications in relevant disciplines such as AI and materials science.
  - High-impact publications by citation count and journal impact factor.
  - Non-peer-reviewed scholarly publications, including arXiv submissions.
  - Other scholarly outputs like conference proceedings.
  - Estimations of new compositions of matter, AI algorithms, and datasets shared by members.

### Implemented Features:

- Notification when an AC member publishes a paper.
- Identifies topics of given papers using a set of keywords.
- Retrieval-Augmented Generation (RAG) from given papers to extract data (currently in progress).
- Only working on last publications of the AC members


**LLM Literature Scraping**

- Define and focus on publications in relevant disciplines by Acceleration Consortium Members.
- **Relevant Disciplines Include:**
  - AI
  - Accelerated materials discovery
  - Self-Driving Labs (SDLs)
  - Autonomous labs
  - High-throughput experimentation
  - High-throughput Density Functional Theory (DFT), etc.
- Scrape the number of publications in peer-reviewed (non-arXiv) literature.
- Calculate the annual average over the last 5 years.

## High Impact Publications

### By Citation Count
- Identify publications in relevant disciplines that are in the top 10% by citation count across all journals.
- Calculate the annual average over the last 5 years.

### By Journal Impact Factor
- Identify publications in relevant disciplines that are in the top 10% by journal impact factor.
- Calculate the annual average over the last 5 years.

### Non-Peer-Reviewed Scholarly Publications

- Count non-peer-reviewed scholarly publications in relevant disciplines, including those on arXiv.
- Aim to filter out papers that are on both arXiv and published in journals.
- Calculate the annual average over the last 5 years.

### Other Scholarly Creations or Outputs

- Include scholarly outputs that do not fit typical publication categories, such as conference proceedings.
- Identify any other outputs found by LLM that don't fit into the above categories.
- Calculate the annual average over the last 5 years.

### New Compositions of Matter
- Estimate the number of new materials identified in patent applications or published literature by Acceleration Consortium Members.
- Provide a best estimate or a range for these numbers.
- Calculate either the annual average over the last 5 years or a total over the previous 5 years.

### New AI or Screening Algorithms, Methods, Workflows, Models
- Estimate the number of new AI/screening algorithms, methods, workflows, or models identified in patent applications or published literature by Acceleration Consortium Members.
- Provide a best estimate or a range for these numbers.
- Calculate either the annual average over the last 5 years or a total over the previous 5 years.

### AI and Material Related Datasets Shared
- Estimate the number of datasets related to AI and materials released to the public by Acceleration Consortium Members.
- Provide a best estimate or a range for these numbers.
- Calculate either the annual average over the last 5 years or a total over the previous 5 years.


## Code Overview

### Author object:
- Takes as input the name of an AC author
- Has the ability to fetch their last publication
  - Uses the scholarly API to get the data (filled)
- Updates a json file with the data it had about the publication

### The Publication object:
- Takes as input a publication filled (output of a scholarly)
- Gets data such as the year of publication, abstract, link etc...
- Uses a LLM on the abstract to get the topics of the publication (Updating it)
- Uses webscrapping of the scholarly home page of the author to get the link to the PDF of the publication (if found - Working on it)

### RAG use - still working on it:
- Embedding used: fast emnbedding (updating it to Ollama's)
- Getting pertinent information from the PDF


### Bot object:
- Runs a Slack bot which is able to send messages and respond to slack slash(/) commands 

