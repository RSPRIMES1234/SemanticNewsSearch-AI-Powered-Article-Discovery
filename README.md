# SemanticNewsSearch-AI-Powered-Article-Discovery
SemanticNewsSearch
SemanticNewsSearch is an AI-powered tool that enables intelligent discovery of news articles based on user queries. It leverages advanced NLP techniques to provide context-aware article recommendations.

Features
* Utilizes the Hugging Face 'multi_news' dataset
* Implements sentence transformers (all-MiniLM-L6-v2) for semantic understanding
* Uses cosine similarity for accurate article matching
* Offers an interactive command-line interface
* Provides concise summaries of top relevant articles
Installation
Clone this repository:

Copy
git clone https://github.com/RSPRIMES1234/SemanticNewsSearch-AI-Powered-Article-Discovery
Install the required packages:

Copy
pip install datasets sentence-transformers torch numpy pandas
Usage
Run the script:


Copy
python semantic_news_search.py
Follow the prompts to enter your search queries. Type 'exit' to quit the program.

How It Works
Loads and preprocesses the 'multi_news' dataset
Encodes article summaries using a pre-trained sentence transformer
Converts user queries into embeddings
Calculates cosine similarity between query and article embeddings
Returns the most relevant articles based on similarity scores
