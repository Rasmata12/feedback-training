import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, pipeline

LABELS = ["AVIS", "RECLAMATION", "SUGGESTION", "EXPERIENCE"]
MARQUEURS_CONTRASTE = ["mais", "cependant", "toutefois", "par contre", "néanmoins"]

tokenizer = XLMRobertaTokenizer.from_pretrained("model_feedback_type_final")
model_type = XLMRobertaForSequenceClassification.from_pretrained("model_feedback_type_final")
model_type.eval()

sentiment_pipe = pipeline("text-classification", model="ac0hik/Sentiment_Analysis_French")

def sentiment_avec_contraste(texte):
    texte_lower = texte.lower()
    for marqueur in MARQUEURS_CONTRASTE:
        if f" {marqueur} " in texte_lower:
            partie_apres = texte.split(marqueur, 1)[1].strip()
            return sentiment_pipe(partie_apres)[0]["label"]
    return sentiment_pipe(texte)[0]["label"]

def analyser_feedback(texte):
    enc = tokenizer(texte, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    with torch.no_grad():
        logits = model_type(**enc).logits
    probs = torch.sigmoid(logits).squeeze().tolist()
    types_predits = [LABELS[i] for i, p in enumerate(probs) if p > 0.5]

    sentiment_brut = sentiment_avec_contraste(texte)

    return {
        "texte": texte,
        "type": types_predits,
        "sentiment": sentiment_brut,
    }

if __name__ == "__main__":
    exemple = analyser_feedback("Le service est bon mais j'ai été débité deux fois.")
    print(exemple)
