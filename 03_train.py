import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments

LABELS = ["AVIS", "RECLAMATION", "SUGGESTION", "EXPERIENCE"]
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

class FeedbackDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = tokenizer(str(row["texte"]), truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor([row[l] for l in LABELS], dtype=torch.float)
        return item

train_ds = FeedbackDataset("data/train.csv")
val_ds = FeedbackDataset("data/val.csv")

model = XLMRobertaForSequenceClassification.from_pretrained(
    "xlm-roberta-base", num_labels=len(LABELS), problem_type="multi_label_classification"
)

args = TrainingArguments(
    output_dir="model_feedback_type",
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    logging_steps=20,
    load_best_model_at_end=True,
)

trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds)
trainer.train()
trainer.save_model("model_feedback_type_final")
tokenizer.save_pretrained("model_feedback_type_final")
print("Entraînement terminé.")
