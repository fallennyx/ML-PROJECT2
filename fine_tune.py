fine_tune.py
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from datasets import Dataset
import pandas as pd
import torch


def fine_tune_model(model_name, preprocessed_data_path, tokenized_data_path, num_epochs=3):
    """Fine-tunes a model on preprocessed and tokenized data."""

    try:
        preprocessed_datasets = pd.read_pickle(preprocessed_data_path)
        tokenized_datasets = pd.read_pickle(tokenized_data_path)
    except FileNotFoundError:
        print("Error loading preprocessed or tokenized data.")
        return

    for dataset_name, data in tokenized_datasets.items():
        print(f"Fine-tuning on dataset: {dataset_name}")
        try:
            # ... Label generation (similar to previous example, create dummy or load existing labels)
            labels = list(range(len(preprocessed_datasets[dataset_name])))
            preprocessed_datasets[dataset_name]["label"] = labels[:len(preprocessed_datasets[dataset_name])]

            dataset = Dataset.from_pandas(preprocessed_datasets[dataset_name])
            dataset = dataset.add_column("input_ids", data['input_ids'])
            dataset = dataset.add_column("attention_mask", data['attention_mask'])

            # Load pre-trained model and tokenizer (use original pre-trained weights)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(set(labels)))
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # ... (Training Arguments and Trainer setup remain the same) ...

            trainer.train()
            trainer.save_model(f"./fine_tuned_model_{dataset_name}")
            print(f"Fine-tuning complete for {dataset_name}. Model saved.")


        except Exception as e:
            print(f"Error fine-tuning on {dataset_name}: {e}")


if __name__ == "__main__":
    model_name = "bert-base-uncased"
    preprocessed_data_path = "preprocessed_data.pkl"  # file paths from where you save the pickle file in preprocess_and_tokenize.py
    tokenized_data_path = "tokenized_data.pkl"
    fine_tune_model(model_name, preprocessed_data_path, tokenized_data_path, num_epochs=1)  # Adjust num_epochs
