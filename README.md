# MLOps Assignment 2 — Hugging Face Fine-Tuning, Experiment Tracking and Model Deployment

**PGD AI Program | IIT Jodhpur | MLOps Course**

---

## Project Description

This project demonstrates a complete MLOps pipeline built around fine-tuning a pre-trained language model for multi-class text classification. The task is genre classification on the UCSD Goodreads dataset, where book review text is classified into literary genre categories. The workflow covers loading a pre-trained DistilBERT model from the Hugging Face Hub, fine-tuning it on a Kaggle GPU, tracking all experiments with Weights and Biases, evaluating the trained model on a held-out test set, and publishing the final model weights and tokenizer back to the Hugging Face Hub. The focus of this assignment is on the operational workflow around the model rather than on the model internals — demonstrating reproducible training, structured experiment tracking, and model publishing as core MLOps competencies.

---

## Project Links

| Resource | Link |
|---|---|
| Hugging Face Model | https://huggingface.co/srajam696/distilbert-goodreads-genres |
| W&B Project Dashboard | https://wandb.ai/srajam696-charan/mlops-assignment2 |
| Kaggle Notebook | https://www.kaggle.com/code/omshivamnlr/mlops2 |

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
└── notebook/
    └── mlops2.ipynb          # Main Kaggle notebook (also linked above)
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Install Dependencies

Python 3.8 or higher is required.

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys

Create a `.env` file in the project root (never commit this file):

```
WANDB_API_KEY=your_wandb_api_key_here
HF_TOKEN=your_huggingface_token_here
```

Alternatively, export them as environment variables:

```bash
export WANDB_API_KEY=your_wandb_api_key_here
export HF_TOKEN=your_huggingface_token_here
```

If running on Kaggle, add these as Secrets via **Add-ons > Secrets** and retrieve them with:

```python
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
WANDB_API_KEY = secrets.get_secret("WANDB_API_KEY")
HF_TOKEN = secrets.get_secret("HF_TOKEN")
```

### 4. Run Training (Kaggle Notebook)

The training was performed on a Kaggle Notebook with a free-tier NVIDIA T4 GPU. To reproduce:

1. Go to [kaggle.com](https://www.kaggle.com) and open the public notebook linked above.
2. Click **Copy and Edit**.
3. Under **Settings**, set **Accelerator** to **GPU T4 x2** and enable **Internet**.
4. Add your `WANDB_API_KEY` and `HF_TOKEN` under **Add-ons > Secrets**.
5. Run all cells end-to-end.

### 5. Run Inference

Load the published model directly from the Hugging Face Hub:

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="srajam696/distilbert-goodreads-genres"
)

result = classifier("A gripping detective story set in Victorian London with dark atmosphere.")
print(result)
```

Or load the model and tokenizer manually:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("srajam696/distilbert-goodreads-genres")
model = AutoModelForSequenceClassification.from_pretrained("srajam696/distilbert-goodreads-genres")

inputs = tokenizer("A young wizard discovers his magical heritage.", return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    logits = model(**inputs).logits

predicted_class = logits.argmax(-1).item()
print(f"Predicted label ID: {predicted_class}")
```

---

## Model Details

| Property | Value |
|---|---|
| Base Model | distilbert-base-cased |
| Task | Multi-class Text Classification (Genre) |
| Dataset | UCSD Goodreads |
| Parameters | 65.8M |
| Training Platform | Kaggle (NVIDIA T4 GPU) |
| Training Epochs | 3 |
| Learning Rate | 3e-5 |
| Train Batch Size | 16 |

---

## Results

| Metric | Score |
|---|---|
| Accuracy | 0.82 |
| Weighted F1 Score | 0.81 |
| Evaluation Loss | 0.61 |

Metrics were logged to the W&B project dashboard and are reproducible by running the public Kaggle notebook with the same hyperparameter configuration.

---

## Requirements

`requirements.txt` contents:

```
torch>=2.0.0
transformers>=4.40.0
datasets>=2.19.0
scikit-learn>=1.3.0
wandb>=0.17.0
huggingface-hub>=0.22.0
```

---

## Acknowledgements

- Dataset: UCSD Goodreads Book Reviews — [McAuley Lab](https://cseweb.ucsd.edu/~jmcauley/)
- Pre-trained model: [distilbert-base-cased](https://huggingface.co/distilbert-base-cased) by Hugging Face
- Training infrastructure: [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks) (free GPU)
- Experiment tracking: [Weights and Biases](https://wandb.ai)
- Model hosting: [Hugging Face Hub](https://huggingface.co)

---

*IIT Jodhpur — PGD AI Program — MLOps Assignment 2*
