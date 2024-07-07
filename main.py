# Import required libraries
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import os

# Load the multi_news dataset from Hugging Face and take only the 'test' split for efficiency
dataset = load_dataset("multi_news", split="test")

# Convert the test dataset to a pandas DataFrame and take only 2000 random samples
df = dataset.to_pandas().sample(2000, random_state=42)

# Load a pre-trained sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode each summary in the DataFrame using the sentence transformer model and store the embeddings in a list
embeddings_list = model.encode(df['summary'].to_list(), show_progress_bar=True)

# Convert the list of NumPy arrays to a single NumPy array
embeddings_array = np.array(embeddings_list)

# Convert the resulting NumPy array to a PyTorch tensor
passage_embeddings = torch.tensor(embeddings_array)

# Print the shape of the first passage embedding
print(passage_embeddings[0].shape)

# Declare a query string
query = "Find me some articles about technology and artificial intelligence"

query_embedding = model.encode(query)
query_embedding_tensor = torch.tensor(query_embedding)
similarities = util.cos_sim(query_embedding_tensor, passage_embeddings)

top_indices = torch.topk(similarities.flatten(), 3).indices
top_relevant_passages = [df.iloc[x.item()]['summary'][:200] + "..." for x in top_indices]
print(top_relevant_passages)

# Define a function to find relevant news articles based on a given query
def find_relevant_news(query):
    # Encode the query using the sentence transformer model
    query_embedding = model.encode(query)
    query_embedding_tensor = torch.tensor(query_embedding)

    # Calculate the cosine similarity between the query embedding and the passage embeddings
    similarities = util.cos_sim(query_embedding_tensor, passage_embeddings)

    # Find the indices of the top 3 most similar passages
    top_indices = torch.topk(similarities.flatten(), 3).indices

    # Get the top 3 relevant passages by slicing the summaries at 200 characters and adding an ellipsis
    top_relevant_passages = [df.iloc[x.item()]['summary'][:200] + "..." for x in top_indices]

    # Return the top 3 relevant passages
    return top_relevant_passages

# Find relevant news articles for different queries (testing the function)

#These are for example


# print(find_relevant_news("Natural disasters"))
# print(find_relevant_news("Law enforcement and police"))
# print(find_relevant_news("Politics, diplomacy and nationalism"))

# Define a function to clear the console screen after each search
def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

# Define a function for people to search news articles by keywords
def interactive_search():
    print("Welcome to the Semantic News Search!\n")
    while True:
        print("Type in a topic you'd like to find articles about, and I'll do the searching! (Type 'exit' to quit)\n> ", end="")

        query = input().strip()

        if query.lower() == "exit":
            print("\nThanks for using the Semantic News Search! Have a great day!")
            break

        print("\n\tHere are 3 articles I found based on your query: \n")

        passages = find_relevant_news(query)
        for passage in passages:
            print("\n\t" + passage)

        input("\nPress Enter to continue searching...")
        clear_screen()

# Start the interactive search
interactive_search()
