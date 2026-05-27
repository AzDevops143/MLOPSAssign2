import json
import wandb
from sklearn.metrics import classification_report
from huggingface_hub import login
from utils import id2label

# Author: g25ait2026, charantej
# Evaluation and deployment module for MLOps Assignment 2

def evaluate_and_push(trainer, test_dataset, tokenizer, hf_repo_name="g25ait2026/distilbert-goodreads-genres"):
    # Run evaluation
    print("Evaluating model...")
    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)
    
    # Log final metrics to W&B
    wandb.log({
        "final/loss":     eval_results.get("eval_loss", 0),
        "final/accuracy": eval_results.get("eval_accuracy", 0),
        "final/f1":       eval_results.get("eval_f1", 0),
    })
    
    # Save full classification report
    print("Generating classification report...")
    preds  = trainer.predict(test_dataset).predictions.argmax(-1)
    labels = [item["labels"].item() for item in test_dataset]
    report = classification_report(labels, preds, target_names=list(id2label.values()), output_dict=True)
    
    with open("eval_report.json", "w") as f:
        json.dump(report, f, indent=2)
        
    # Upload to W&B as a versioned Artifact
    artifact = wandb.Artifact("eval-report", type="evaluation")
    artifact.add_file("eval_report.json")
    wandb.log_artifact(artifact)
    
    # Push to Hugging Face
    hf_token = None
    try:
        with open("APIkey.txt", "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            for i, line in enumerate(lines):
                if line == "HFAPI" and i - 1 >= 0:
                    hf_token = lines[i-1] # Based on APIkey.txt format: hf_... \n HFAPI
                    break
            if not hf_token and len(lines) >= 1:
                if lines[0].startswith("hf_"):
                    hf_token = lines[0]
                    
        if hf_token:
            print("Logging into Hugging Face...")
            login(token=hf_token)
            
            print(f"Pushing model and tokenizer to {hf_repo_name}...")
            trainer.model.push_to_hub(hf_repo_name)
            tokenizer.push_to_hub(hf_repo_name)
            
            wandb.run.summary["huggingface_model"] = f"https://huggingface.co/{hf_repo_name}"
            print("Successfully pushed to Hugging Face Hub.")
        else:
            print("HF Token not found, skipping push to Hub.")
    except Exception as e:
        print(f"Error during Hugging Face push: {e}")
        
    wandb.finish()
    print("Evaluation and deployment complete.")

if __name__ == "__main__":
    print("This script is meant to be called after training.")
