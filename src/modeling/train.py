from pathlib import Path

import torch
import typer
import ast
import pandas as pd
from loguru import logger

from torch import nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from src.config import PROCESSED_DATA_DIR, loss_func

app = typer.Typer()



class LawDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        question = row["question_body"]
        answers = ast.literal_eval(row["answers"])
        if len(answers) == 0:
            return {
            "question": question,
            "positive": "",
            "negative": ""
        }
        max_idx = 0
        max_score = answers[0]["score"]
        min_idx = 0
        min_score = answers[0]["score"]
        for i, ans in enumerate(answers):
            tmp_score = ans['score']
            if max_score < tmp_score:
                max_score = tmp_score
                max_idx = i
            if min_score > tmp_score:
                min_score = tmp_score
                min_idx = i

        positive_answer = answers[max_idx]["body"] if max_score >= 0 else ""

        negative_answer = answers[min_idx]["body"] if min_score < 0 else ""


        return {
            "question": question,
            "positive": positive_answer,
            "negative": negative_answer
        }


@app.command()
def train(
    test_path: Path = PROCESSED_DATA_DIR / "test_processed_dataset.csv",
    train_path: Path = PROCESSED_DATA_DIR / "train_processed_dataset.csv"
):
    logger.info("Start training model")
    model_name = "intfloat/e5-small-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = LawDataset(pd.read_csv(train_path), tokenizer)
    test_dataset = LawDataset(pd.read_csv(test_path), tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)


    class CollateFn:
        def __init__(self, tokenizer, max_length=128):
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __call__(self, batch):
            questions = batch["question"]
            positives = batch["positive"]
            negatives = batch["negative"]


            tokenized_questions = self.tokenizer(questions, padding=True, truncation=True, max_length=self.max_length,
                                                 return_tensors="pt")
            tokenized_positives = self.tokenizer(positives, padding=True, truncation=True, max_length=self.max_length,
                                                 return_tensors="pt")
            tokenized_negatives = self.tokenizer(negatives, padding=True, truncation=True, max_length=self.max_length,
                                                 return_tensors="pt")

            return tokenized_questions, tokenized_positives, tokenized_negatives

    collate_fn = CollateFn(tokenizer)


    # Model setup
    class SimilarityModel(nn.Module):
        def __init__(self, model_name):
            super(SimilarityModel, self).__init__()
            self.encoder = AutoModel.from_pretrained(model_name)

        def forward(self, input_ids, attention_mask):
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token embeddings
            return embeddings

    model = SimilarityModel(model_name)


    def compute_similarity(embeddings1, embeddings2):
        return torch.cosine_similarity(embeddings1, embeddings2)

    # Loss functions
    def triplet_margin_loss(anchor, positive, negative, margin=1.0):
        pos_distance = 1 - compute_similarity(anchor, positive)
        neg_distance = 1 - compute_similarity(anchor, negative)
        return torch.relu(pos_distance - neg_distance + margin).mean()

    def contrastive_loss(anchor, positive, negative, margin=1.0):
        pos_distance = (1 - compute_similarity(anchor, positive)) ** 2
        neg_distance = torch.relu(margin - compute_similarity(anchor, negative)) ** 2
        return (pos_distance + neg_distance).mean()

    # Training loop
    def train_model(model, train_loader, loss_fn, optimizer, epochs=3):
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                batch = collate_fn(batch)
                optimizer.zero_grad()
                questions, positives, negatives = batch
                question_embeddings = model(questions["input_ids"], questions["attention_mask"])
                positive_embeddings = model(positives["input_ids"], positives["attention_mask"])
                negative_embeddings = model(negatives["input_ids"], negatives["attention_mask"])

                loss = loss_fn(question_embeddings, positive_embeddings, negative_embeddings)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

    # Optimizer
    optimizer = Adam(model.parameters(), lr=2e-3)

    # Train with Triplet Margin Loss
    logger.info("Training with Triplet Margin Loss")
    get_loss = {"triplet_margin_loss": triplet_margin_loss,
                "contrastive_loss": contrastive_loss}
    train_model(model, train_loader, get_loss[loss_func], optimizer)

    # Evaluate model on test data (simple similarity metric)
    def evaluate_model(model, test_loader):
        logger.info("Start evaluation")
        model.eval()
        total_similarity = 0
        count = 0

        with torch.no_grad():
            for batch in test_loader:
                questions, positives, _ = batch
                question_embeddings = model(questions["input_ids"], questions["attention_mask"])
                positive_embeddings = model(positives["input_ids"], positives["attention_mask"])

                similarities = compute_similarity(question_embeddings, positive_embeddings)
                total_similarity += similarities.sum().item()
                count += len(similarities)

        return total_similarity / count

    similarity_score = evaluate_model(model, test_loader)

    logger.success(f"Modeling training complete. \n Similarity Score: {similarity_score}")


if __name__ == "__main__":
    app()
