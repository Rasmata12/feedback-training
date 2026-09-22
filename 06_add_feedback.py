import pandas as pd
from importlib import import_module
analyse = import_module("05_analyse_complete")

nouveau_texte = "Le livreur est arrivé avec une heure de retard."

df = pd.read_csv("data/base_ikan.csv")

if nouveau_texte in df["texte"].values:
    print("Ce feedback existe déjà dans la base, rien ajouté.")
else:
    resultat = analyse.analyser_feedback(nouveau_texte)
    nouvelle_ligne = {
        "texte": nouveau_texte,
        "AVIS": 1 if "AVIS" in resultat["type"] else 0,
        "RECLAMATION": 1 if "RECLAMATION" in resultat["type"] else 0,
        "SUGGESTION": 1 if "SUGGESTION" in resultat["type"] else 0,
        "EXPERIENCE": 1 if "EXPERIENCE" in resultat["type"] else 0,
        "source": "pilote_client",
        "valide_humain": 0,
    }
    df = pd.concat([df, pd.DataFrame([nouvelle_ligne])], ignore_index=True)
    df.to_csv("data/base_ikan.csv", index=False)
    print("Feedback ajouté (type), en attente de validation humaine.")
    print("Sentiment détecté (info) :", resultat["sentiment"])
