import pandas as pd
import pinecone
import openai
from langchain.embeddings import OpenAIEmbeddings

# Initialize Pinecone
pinecone.init(api_key='your_pinecone_api_key', environment='your_pinecone_environment')

# Create or connect to an existing Pinecone index
index_name = 'parties'
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name)
index = pinecone.Index(index_name)

# Initialize OpenAI embeddings
model_name = 'text-embedding-ada-002'
embeddings = OpenAIEmbeddings(model=model_name, openai_api_key='your_openai_api_key')

# Assume a DataFrame 'parties_df' with columns 'party_id', 'description', 'location', 'date'
parties_df = pd.read_csv('path_to_parties.csv')

# Vectorize party descriptions and upsert into Pinecone
for _, row in parties_df.iterrows():
    vector = embeddings.embed_query(row['description'])
    index.upsert(vectors={(row['party_id']): vector})

# User query
user_query = "Halloween party near New York on October 31"
query_vector = embeddings.embed_query(user_query)

# Perform the search
results = index.query(queries=[query_vector], top_k=5, include_metadata=False)

# Retrieve party details based on results
party_ids = [match['id'] for match in results['results'][0]['matches']]
matched_parties = parties_df[parties_df['party_id'].isin(party_ids)]

print(matched_parties)
