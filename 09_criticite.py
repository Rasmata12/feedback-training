SIGNAUX_URGENCE = [
    "urgent", "immédiatement", "tout de suite", "depuis plusieurs jours",
    "depuis des semaines", "toujours pas", "aucune réponse", "personne ne répond",
]
SIGNAUX_PREJUDICE = [
    "débité", "remboursement", "argent", "facturé", "payé deux fois",
    "perdu", "vol", "sécurité", "danger", "blessé",
]
NIVEAUX = ["faible", "moyenne", "elevee", "critique"]

def detecter_signaux(texte):
    texte_lower = texte.lower()
    urgence = [m for m in SIGNAUX_URGENCE if m in texte_lower]
    prejudice = [m for m in SIGNAUX_PREJUDICE if m in texte_lower]
    return urgence, prejudice

def compute_criticite(note, sentiment, texte=""):
    if sentiment == "negative" and note <= 2:
        niveau = "critique"
    elif sentiment == "negative":
        niveau = "elevee"
    elif sentiment == "neutral":
        niveau = "moyenne"
    else:
        niveau = "faible"

    urgence, prejudice = detecter_signaux(texte)
    if (urgence or prejudice) and NIVEAUX.index(niveau) < NIVEAUX.index("elevee"):
        niveau = "elevee"
    if urgence and prejudice and niveau != "critique":
        niveau = "critique"

    return niveau, urgence, prejudice
