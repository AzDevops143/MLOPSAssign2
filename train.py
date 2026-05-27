import wandb
import os
import torch
from transformers import TrainingArguments, Trainer, DistilBertForSequenceClassification
from utils import id2label, compute_metrics

# Author: g25ait2026, charantej
# Training module for MLOps Assignment 2

def train_model(train_dataset, test_dataset, model_name='distilbert-base-cased', max_length=512):
    # Try to load API key
    try:
        with open("APIkey.txt", "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            wandb_key = None
            for i, line in enumerate(lines):
                if line == "wandbapi" and i + 1 < len(lines):
                    wandb_key = lines[i+1]
                    break
            if not wandb_key and len(lines) >= 4:
                # Fallback if specific line not found
                wandb_key = lines[-1]
                
        if wandb_key:
            wandb.login(key=wandb_key)
    except Exception as e:
        print(f"Could not load W&B key: {e}")
        
    wandb.init(
        project="mlops-assignment2",
        name="distilbert-run-charantej",
        config={
            "model": model_name,
            "epochs": 3,
            "batch_size": 16,
            "learning_rate": 3e-5,
            "max_length": max_length,
            "dataset": "UCSD Goodreads",
            "student": "g25ait2026,charantej"
        }
    )
    
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device_name}")
    
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(id2label)
    ).to(device_name)
    
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb",
        run_name="distilbert-run-charantej",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("Starting training...")
    trainer.train()
    
    return trainer, model

if __name__ == "__main__":
    from data import prepare_data
    train_ds, test_ds, tok, texts, labels = prepare_data(sample_size=200) # Small sample for testing
    trainer, model = train_model(train_ds, test_ds)
    wandb.finish()
