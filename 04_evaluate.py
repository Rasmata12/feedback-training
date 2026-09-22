import pandas as pd
import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from sklearn.metrics import classification_report

LABELS = ["AVIS", "RECLAMATION", "SUGGESTION", "EXPERIENCE"]
tokenizer = XLMRobertaTokenizer.from_pretrained("model_feedback_type_final")
model = XLMRobertaForSequenceClassification.from_pretrained("model_feedback_type_final")
model.eval()

df = pd.read_csv("data/test.csv")
preds, trues = [], []

for _, row in df.iterrows():
    enc = tokenizer(str(row["texte"]), truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.sigmoid(logits).squeeze().tolist()
    pred = [1 if p > 0.5 else 0 for p in probs]
    preds.append(pred)
    trues.append([row[l] for l in LABELS])

print(classification_report(trues, preds, target_names=LABELS, zero_division=0))
