import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch

# Tokenize descriptions manually
def simple_tokenizer(text):
    return text.lower().split()

# Function to preprocess the IMDb dataset for recommendation model
def preprocess_data(basics_path, ratings_path):
    # Load datasets without specifying data types initially
    title_basics = pd.read_csv(basics_path, sep='\t', na_values='\\N', low_memory=False)
    title_ratings = pd.read_csv(ratings_path, sep='\t', na_values='\\N')

    # Convert 'runtimeMinutes' to numeric, forcing errors to NaN
    title_basics['runtimeMinutes'] = pd.to_numeric(title_basics['runtimeMinutes'], errors='coerce')

    # Convert 'startYear' and 'endYear' to numeric, forcing errors to NaN
    title_basics['startYear'] = pd.to_numeric(title_basics['startYear'], errors='coerce')
    title_basics['endYear'] = pd.to_numeric(title_basics['endYear'], errors='coerce')

    # Drop rows with NaNs in key columns
    df = pd.merge(title_basics, title_ratings, on='tconst', how='left')
    df = df.dropna(subset=['primaryTitle', 'genres', 'averageRating', 'runtimeMinutes'])

    # Normalize numerical features
    scaler = MinMaxScaler()
    df[['averageRating', 'runtimeMinutes']] = scaler.fit_transform(df[['averageRating', 'runtimeMinutes']])

    # Create a combined text for tokenization
    df['combined_text'] = df['primaryTitle'] + ' ' + df['originalTitle'] + ' ' + df['genres']

    # Tokenize the combined text
    df['description'] = df['combined_text'].apply(lambda x: simple_tokenizer(x))

    # Build a vocabulary from the descriptions
    vocab = set()
    for desc in df['description']:
        vocab.update(desc)
    vocab = {word: idx + 1 for idx, word in enumerate(vocab)}  # start indexing from 1

    # Convert descriptions to sequences of indices
    df['description'] = df['description'].apply(lambda desc: [vocab[word] for word in desc])

    # Padding sequences
    max_seq_len = max(df['description'].apply(len))
    df['description'] = df['description'].apply(lambda x: x + [0] * (max_seq_len - len(x)))

    # Encode primaryTitle for numerical operations
    label_encoder = LabelEncoder()
    df['encoded_primaryTitle'] = label_encoder.fit_transform(df['primaryTitle'])

    return df, vocab, max_seq_len, label_encoder

# Create a custom dataset class
class IMDbDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        return {
            'movie_name': torch.tensor(row['encoded_primaryTitle'], dtype=torch.long),
            'rating': torch.tensor(row['averageRating'], dtype=torch.float),
            'runtime': torch.tensor(row['runtimeMinutes'], dtype=torch.float),
            'description': torch.tensor(row['description'], dtype=torch.long),
        }

# Function to create DataLoader
def create_dataloader(df, batch_size=16, shuffle=True, num_workers=0):
    dataset = IMDbDataset(df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

# Function to preprocess data specifically for recommendation model
def preprocess_for_recommendation(df):
    # Assuming you have user_id, movie_id, and rating columns for the recommendation model
    # Create a dummy user_id column for this example, replace with actual user_id in your dataset
    df['user_id'] = range(1, len(df) + 1)

    recommendation_df = df[['user_id', 'encoded_primaryTitle', 'averageRating']].rename(
        columns={'encoded_primaryTitle': 'movie_id', 'averageRating': 'rating'}
    )

    return recommendation_df

"""
# Get absolute paths to data files
base_path = '/Users/wenhaohe/Desktop/MLE/data/'
basics_path = os.path.join(base_path, 'title.basics.tsv')
ratings_path = os.path.join(base_path, 'title.ratings.tsv')

# Preprocess data
df, vocab, max_seq_len, label_encoder = preprocess_data(basics_path, ratings_path)
dataloader = create_dataloader(df)

# Print some details
print(df.head())
print(f"Vocabulary size: {len(vocab)}")
print(f"Max sequence length: {max_seq_len}")
"""
