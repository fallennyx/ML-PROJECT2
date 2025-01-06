from datasets import load_dataset
import pandas as pd

# Load dataset from Hugging Face
dataset = load_dataset("enryu43/twitter100m_tweets")

# Check if the dataset is loaded correctly
print(f"Dataset loaded: {dataset}")
print(f"Dataset keys: {dataset.keys()}")  # Should show dataset splits like 'train', 'test', etc.

# Inspect the first few rows of a specific split
df = dataset["train"].to_pandas()
print(df.head())  # Display the first 5 rows
