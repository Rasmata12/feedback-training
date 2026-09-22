from importlib import import_module
analyse = import_module("05_analyse_complete")
discordance_mod = import_module("08_discordance")
criticite_mod = import_module("09_criticite")

def analyser_feedback_complet(texte, note):
    base = analyse.analyser_feedback(texte)
    criticite, urgence, prejudice = criticite_mod.compute_criticite(note, base["sentiment"], texte)
    discordance = discordance_mod.detect_discordance(note, base["sentiment"])

    base["criticite"] = criticite
    base["discordance"] = discordance
    base["signaux_urgence"] = urgence
    base["signaux_prejudice"] = prejudice
    return base

if __name__ == "__main__":
    exemple = analyser_feedback_complet(
        "Le service est bon mais j'ai été débité deux fois et personne ne répond.",
        note=5,
    )
    print(exemple)
