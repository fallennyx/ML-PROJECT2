# Import necessary libraries
from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd

# === Step 1: Define datasets and their corresponding text columns === #
# These are the datasets and the specific text column containing the tweet data
#abt 92.5 million tweets
DATASETS = {
    "globalgreen/tweet_smileys": "text",  # Text column is "text"
    "enryu43/twitter100m_tweets": "tweet",  # Text column is "tweet"
    "ad321/test-tweets": "text"  # Text column is "text"
}

# === Step 2: Initialize tokenizer === #
# Using the tokenizer for a pre-trained model (BERT in this case)
MODEL_NAME = "bert-base-uncased"
print(f"Loading tokenizer for model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("Tokenizer loaded successfully.\n")

# === Step 3: Consolidated Preprocessing
# Function === #
def preprocess_data(dataset, text_column):
    """
    Preprocesses the dataset, including URL, mention, hashtag, and special character removal.

    Args:
      dataset: Hugging Face dataset.
      text_column: Name of the column containing the text data.

    Returns:
      Preprocessed Pandas DataFrame.
    """
    print(f"Preprocessing dataset '{dataset.name}', focusing on '{text_column}' column...")
    df = dataset.to_pandas()

    # Handle missing values:
    df = df.dropna(subset=[text_column])
    df = df[df[text_column].str.strip() != ""]

    # Apply all preprocessing steps in one go:
    df[text_column] = df[text_column].astype(str).apply(
        lambda text: text.lower().strip()
        .replace('http\S+|www\S+', '', regex=True)
        .replace('@\w+|#\w+', '', regex=True)
        .replace('[^A-Za-z\s]', '', regex=True)
    )

    print(f"Preprocessing completed: {len(df)} entries remaining.\n")
    return df

# === Step 4: Define a function to tokenize the data === #
def tokenize_data(df, text_column):
    """
    Tokenize the text data in the DataFrame using the Hugging Face tokenizer.

    Parameters:
    - df: Preprocessed DataFrame containing the text data.
    - text_column: The specific column containing the text for tokenization.

    Returns:
    - Dictionary of tokenized input IDs, attention masks, etc.
    """
    print(f"Tokenizing data in column '{text_column}'...")

    # Tokenize text using the Hugging Face tokenizer
    tokenized_data = tokenizer(
        df[text_column].tolist(),  # Hugging Face tokenizer works with lists of text
        truncation=True,  # Truncates text longer than model's maximum length
        padding="max_length",  # Pads shorter texts to the maximum length
        max_length=128  # Maximum length (adjust if needed)
    )

    print("Tokenization completed successfully.\n")
    return tokenized_data


# === Step 5: Loop through datasets, preprocess, and tokenize === #
def process_datasets(datasets, preprocess_func, tokenize_func, batch_size=1000):
    """
    Processes multiple datasets, handling potential memory issues.
    """
    preprocessed_datasets = {}
    tokenized_datasets = {}

    for dataset_name, text_column in datasets.items():
        print(f"Processing dataset: {dataset_name}")
        try:
            dataset = load_dataset(dataset_name)

            if dataset_name == "enryu43/twitter100m_tweets":
                # Process in batches for large dataset
                preprocessed_dfs = []
                for i in range(0, len(dataset["train"]), batch_size):
                    batch = dataset["train"][i:i + batch_size]
                    preprocessed_dfs.append(preprocess_func(batch, text_column))
                preprocessed_df = pd.concat(preprocessed_dfs)

            else:
                # Process normally for smaller datasets
                preprocessed_df = preprocess_func(dataset["train"], text_column)

            preprocessed_datasets[dataset_name] = preprocessed_df
            tokenized_data = tokenize_func(preprocessed_df, text_column)
            tokenized_datasets[dataset_name] = tokenized_data

        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")

    return preprocessed_datasets, tokenized_datasets

if __name__ == "__main__":  # This ensures the code below runs only when directly running this file
    preprocessed_datasets, tokenized_datasets = process_datasets(DATASETS, preprocess_data, tokenize_data)
    print("Saving preprocessed and tokenized datasets...")
    pd.to_pickle(preprocessed_datasets, "preprocessed_data.pkl")
    pd.to_pickle(tokenized_datasets, "tokenized_data.pkl")
    print("Preprocessed and tokenized datasets saved successfully.")