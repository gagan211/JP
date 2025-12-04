# Email vs Calendar Query Classifier

A **binary text classification model** that categorizes user queries into Email (Gmail) or Calendar related categories using fine-tuned **DistilBERT**.

## Overview

This model classifies natural language queries into two categories:

- **Class 0**: Gmail-related queries (emails, attachments, inbox, labels, etc.)
- **Class 1**: Calendar-related queries (meetings, events, appointments, schedules, etc.)

## Model Architecture

- **Base Model**: DistilBERT (distilbert-base-uncased)
- **Framework**: PyTorch
- **Optimizer**: AdamW with linear warmup schedule
- **Learning Rate**: 2e-5
- **Max Sequence Length**: 128 tokens
- **Training Batch Size**: 16
- **Epochs**: 2-3

### Why DistilBERT?

| Feature | Benefit |
|---------|---------|
| **Size** | 66% smaller than BERT-base |
| **Performance** | Maintains ~97% of BERT's accuracy |
| **Speed** | 60% faster inference |
| **Use Case** | Optimal balance for production deployment |

## Performance Metrics

The model is evaluated using:
- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall

## Download the Model

The trained model is available on Kaggle:

### Method 1: Using Kaggle API (Recommended)

1. **Install Kaggle CLI** (if not already installed):
   ```bash
   pip install kaggle
   ```

2. **Set up Kaggle API credentials**:
   - Go to [Kaggle Settings → API](https://www.kaggle.com/settings/account)
   - Click "Create New API Token"
   - Save the `kaggle.json` file to `~/.kaggle/`
   - Set permissions:
     ```bash
     chmod 600 ~/.kaggle/kaggle.json
     ```

3. **Download the model**:
   ```bash
   kaggle kernels output gaganm2001/ml-assessment1 -p /path/to/destination
   ```

### Method 2: Manual Download

1. Visit the Kaggle Kernel: (https://www.kaggle.com/code/gaganm2001/ml-assessment1)
2. Click the **Output** tab
3. Download the model files manually

## Dataset

Training and evaluation datasets are stored in:
- `data/set1/` - Primary dataset (train/val/test split)
  - `mail_calendar_dataset_train.csv`
  - `mail_calendar_dataset_val.csv`
  - `mail_calendar_dataset_test.csv`
- `data/set2/` - Additional dataset
  - `email_calendar_dataset_v2.csv`

## Usage

### Using the Model in Your Code

```python
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch

# Load model and tokenizer
model_name = "distilbert-base-uncased"
model = DistilBertForSequenceClassification.from_pretrained("path/to/model")
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

# Define prediction function
def predict_class(query: str) -> int:
    inputs = tokenizer(query, return_tensors="pt", max_length=128, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return prediction

# Make predictions
query = "Send me my emails"
category = predict_class(query)
print(f"Category: {'Gmail' if category == 0 else 'Calendar'}")
```

## Bonus Features

- Extract time ranges from calendar queries
- Extract people/entities from queries
- Error analysis and stress-testing utilities

## Implementation Details

The notebook includes:
1. **Environment Setup** - Library imports and configuration
2. **Data Loading & Exploration** - Dataset analysis
3. **Data Validation** - Quality checks
4. **Tokenization** - Text preprocessing
5. **Model Training** - Fine-tuning DistilBERT
6. **Evaluation** - Performance metrics on test set
7. **Inference** - Prediction functions
8. **Error Analysis** - Misclassified examples

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.0+
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn (for visualization)

## Installation

```bash
pip install torch transformers kaggle pandas numpy scikit-learn matplotlib seaborn
```

## Notebook Location

The complete model implementation is available at:
- **Kaggle Kernel**: (https://www.kaggle.com/code/gaganm2001/ml-assessment1)
- **Local Notebook**: `/notebook/ml-assessment1.ipynb`

## License

This model is provided as-is for educational and research purposes.

## Author

Created by Gagan
- **Kaggle Profile**: (https://www.kaggle.com/gaganm2001)

---

**Last Updated**: December 2025
