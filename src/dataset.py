from pathlib import Path

import pandas as pd
import typer
from bs4 import BeautifulSoup
from datasets import load_dataset
from loguru import logger
from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, seed

app = typer.Typer()
pd.options.display.max_columns = None


@app.command()
def preprocess_dataset(
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    train_output_path: Path = PROCESSED_DATA_DIR / "train_processed_dataset.csv",
    test_output_path: Path = PROCESSED_DATA_DIR / "test_processed_dataset.csv",
    dataset_name: str = "ymoslem/Law-StackExchange",
):
    """Load, preprocess, and save the dataset."""
    logger.info("Loading dataset...")
    dataset = load_dataset(dataset_name)['train']

    def clean_html(text):
        return BeautifulSoup(text, 'html.parser').get_text()


    def load_and_preprocess_data(raw_dataset):
        logger.info("Cleaning HTML from the dataset...")
        df = pd.DataFrame(raw_dataset)

        # Clean HTML tags in questions and answers
        df["question_body"] = df["question_body"].apply(clean_html)
        df['question_body'] = df.apply(lambda row:
                                       f"query: {row['question_title']} \n {row['question_body']} ",
                                       axis=1)
        df = df.drop(columns=['question_title', 'score', 'license', 'link', 'tags'])
        for idx, answers in enumerate(df["answers"]):
            if answers:
                for ans in answers:
                    ans["body"] = clean_html(ans["body"])
                    ans["body"] = ''.join(("passage: ", ans["body"]))
        return df

    processed_data = load_and_preprocess_data(dataset)
    train_data, test_data = train_test_split(processed_data, test_size=0.2, random_state=seed)
    logger.info("Saving the processed dataset...")
    train_data.to_csv(train_output_path, index=False, encoding='utf-8')
    test_data.to_csv(test_output_path, index=False, encoding='utf-8')
    logger.success("Dataset preprocessing complete.")


if __name__ == "__main__":
    app()
