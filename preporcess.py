import re
import os
import pandas as pd
import nltk
from pathlib import Path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from huggingface_hub import HfApi
from datasets import load_dataset

# Download needed NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Initialize Hugging Face API
# Commented out due to already making repo
# HfApi().create_repo(repo_id="Username/my_dataset", repo_type="dataset")


# Preprocessing function for an individual tweet
def preprocess_tweet(tweet):
    # Removes URLs
    tweet = re.sub(r"http\S+|www\S+", "", tweet)
    # Remove mentions and hashtags
    tweet = re.sub(r"@\w+|#\w+", "", tweet)
    # Remove special characters, numbers, and punctuations
    tweet = re.sub(r"[^A-Za-z\s]", "", tweet)
    # Convert to lowercase
    tweet = tweet.lower()
    # Tokenize
    words = word_tokenize(tweet)
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]
    # Join words back to form the preprocessed tweet
    return " ".join(words)


# Preprocessing function for an entire dataset
def preprocess_dataset(ds, text_column):
    # Convert Hugging Face dataset to Pandas DataFrame
    df = ds.to_pandas()
    # Remove empty or whitespace-only rows
    df = df[df[text_column].str.strip() != ""]
    # Apply tweet preprocessing function
    df[text_column] = df[text_column].apply(preprocess_tweet)
    return df


if __name__ == "__main__":
    # List of datasets and their respective text columns
    datasets_info = {
        "enryu43/twitter100m_tweets": "tweet",  # Column name is "tweet"
        "ad321/test-tweets": "text",  # Column name is "text"
        "globalgreen/tweet_smileys": "input",  # Column name is "input"
    }

    # Initialize an empty list to store processed datasets
    datasets_processed = []

    # Iterate through each dataset in the list
    for dataset_name, text_column in datasets_info.items():
        print(f"Loading dataset: {dataset_name}")
        # Load dataset using Hugging Face's "datasets" library
        ds = load_dataset(dataset_name)
        # Preprocess the dataset's specified column
        df_processed = preprocess_dataset(ds["train"], text_column)
        # Save processed dataframe to the list
        datasets_processed.append(df_processed)

    # Directory to save processed datasets
    save_path = Path("./processed_datasets")
    save_path.mkdir(exist_ok=True)

    # Save processed datasets as CSV files
    for i, dataset_df in enumerate(datasets_processed):
        file_path = save_path / f"dataset_{i + 1}.csv"
        print(f"Saving processed dataset to: {file_path}")
        dataset_df.to_csv(file_path, index=False)

    print("All datasets processed and saved.")
