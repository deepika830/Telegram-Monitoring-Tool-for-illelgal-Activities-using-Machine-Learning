import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import classification_report
import numpy as np
import json

# 1. Load and prepare data
df = pd.read_csv("mydata1.csv")  
df.columns = [col.strip().lower() for col in df.columns]

# Define your category mapping
category_map = {
    0: "normal",
    1: "financial_scam",
    2: "credential_phishing",
    3: "fake_offers",
    4: "urgency_threats",
    5: "otherscams"
}

# 2. Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# 3. Tokenize
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True)

# 4. Dataset Class
class ScamDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ScamDataset(train_encodings, list(train_labels))
val_dataset = ScamDataset(val_encodings, list(val_labels))

# 5. Custom Trainer with Class Weights
class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(model.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
            
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# 6. Model Setup
num_classes = len(category_map)
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", 
    num_labels=num_classes
)

# 7. Enhanced Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    
    target_names = list(category_map.values())
    label_ids = list(category_map.keys())
    
    report = classification_report(
        labels, 
        preds, 
        output_dict=True,
        target_names=target_names,
        labels=[int(k) for k in label_ids]
    )
    return {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"]
    }

# 8. Training Setup
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=6,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    seed=42
)

# Define class weights (adjust these values as needed)
class_weights = torch.tensor([1.0, 3.0, 3.0, 3.0, 3.0, 3.0], dtype=torch.float32)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    class_weights=class_weights  # Pass class weights here
)

# 9. Train and save
trainer.train()

model.save_pretrained("scam_type_detector")
tokenizer.save_pretrained("scam_type_detector")

# Save category mapping for inference
with open("category_map.json", "w") as f:
    json.dump(category_map, f)

print("✅ Model saved with categories:", category_map)