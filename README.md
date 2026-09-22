# Feedback Type Classification

Ce projet vise à classifier automatiquement des retours utilisateurs (feedback) en fonction de leur nature, en français, avec un modèle de classification multi-label. L’objectif est de distinguer les messages de type :

- AVIS
- RECLAMATION
- SUGGESTION
- EXPERIENCE

Le modèle est basé sur XLM-RoBERTa, un modèle multilingue de Transformers, finetuné sur un dataset de feedback utilisateur.

---

## Objectif du projet

Les retours clients peuvent contenir plusieurs enjeux à la fois :
- une remarque générale sur le service,
- une réclamation technique,
- une demande d’évolution,
- ou une expérience utilisateur vécue.

Le système cherche à détecter automatiquement ces catégories afin d’aider à prioriser les réponses, analyser les tendances et mieux comprendre les frustrations des utilisateurs.

---

## Données

Les données sont stockées dans le dossier `data/`.

### Sources

- `data/base_ikan.csv` : base consolidée issue des données de feedback
- `data/apia2022/*.csv` : données d’origine du dataset APIA 2022
- `data/train.csv`, `data/val.csv`, `data/test.csv` : splits train/validation/test

### Structure

Chaque ligne contient un texte d’utilisateur et des colonnes binaire de labels :

- `texte`
- `AVIS`
- `RECLAMATION`
- `SUGGESTION`
- `EXPERIENCE`

Les labels sont binaires (0 ou 1) et un même commentaire peut appartenir à plusieurs catégories simultanément.

---

## Pipeline de traitement

### 1. Préparation des données
`01_prepare_data.py`

- charge les fichiers CSV du dossier `data/apia2022/`
- renomme les colonnes pour harmoniser le format
- supprime les lignes vides et doublons
- normalise les textes
- sauvegarde la base consolidée dans `data/base_ikan.csv`

### 2. Découpage des données
`02_split_data.py`

- découpe le dataset en train / validation / test
- répartition approximative : 80% / 10% / 10%

### 3. Entraînement du modèle
`03_train.py`

- utilise `xlm-roberta-base`
- configure une classification multi-label
- entraîne pendant 4 epochs
- sauvegarde le modèle final dans `model_feedback_type_final`

### 4. Évaluation
`04_evaluate.py`

- charge le meilleur modèle
- applique le seuil de 0.5 sur les logits sigmoid
- affiche le rapport de classification complet

### 5. Analyse avancée
Les scripts suivants complètent le projet :

- `05_analyse_complete.py` : analyse complète du feedback
- `06_add_feedback.py` : ajout ou enrichissement de nouvelles données
- `08_discordance.py` : détection de discordance entre sentiment et type détecté
- `09_criticite.py` : calcul de criticité / signal de risque
- `10_analyse_finale.py` : synthèse de l’analyse finale
- `13_dedupe.py` : déduplication des entrées

---

## Modèle utilisé

- Architecture : `XLM-RoBERTa` (`xlm-roberta-base`)
- Type de tâche : classification multi-label
- Seuil d’activation : 0.5
- Fonction de sortie : sigmoid appliquée sur les logits

---

## Résultats obtenus

L’évaluation a été réalisée sur le jeu de test. Voici les résultats réels obtenus :

| Label | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| AVIS | 0.88 | 0.88 | 0.88 | 298 |
| RECLAMATION | 0.92 | 0.91 | 0.92 | 202 |
| SUGGESTION | 0.86 | 0.88 | 0.87 | 107 |
| EXPERIENCE | 0.82 | 0.75 | 0.78 | 120 |

### Moyennes globales

| Metric | Score |
|---|---:|
| Micro avg | 0.87 |
| Macro avg | 0.86 |
| Weighted avg | 0.87 |
| Samples avg | 0.88 |

### Interprétation

- La catégorie `RECLAMATION` est la mieux détectée avec un F1-score de `0.92`.
- `AVIS` est très stable avec un F1-score de `0.88`.
- `SUGGESTION` est correcte avec `0.87`.
- `EXPERIENCE` est plus difficile à classifier, avec `0.78`, probablement à cause d’un contexte plus subjectif et variable.

Le score global moyen macro est de `0.86`, ce qui indique une bonne qualité de classification sur les 4 catégories.

---

## Réplication rapide

### Prérequis

Python 3.11 et les dépendances suivantes :

```bash
pip install torch transformers pandas scikit-learn sentencepiece
```

### Commandes

```bash
python 01_prepare_data.py
python 02_split_data.py
python 03_train.py
python 04_evaluate.py
```

---

## Structure du dépôt

```text
feedback_type/
├── 01_prepare_data.py
├── 02_split_data.py
├── 03_train.py
├── 04_evaluate.py
├── 05_analyse_complete.py
├── 06_add_feedback.py
├── 08_discordance.py
├── 09_criticite.py
├── 10_analyse_finale.py
├── 13_dedupe.py
├── data/
│   ├── apia2022/
│   ├── base_ikan.csv
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── model_feedback_type/
├── model_feedback_type_final/
├── .gitignore
├── README.md
└── ...
```

> Les dossiers de modèles sont ignorés dans le dépôt Git pour ne pas uploader les poids de gros modèles. Il faut relancer l’entraînement pour reconstruire le modèle localement.

---

## Conclusion

Ce projet montre qu’un modèle de type XLM-RoBERTa peut bien identifier les principales catégories de feedback utilisateur en français, avec un bon niveau de performance global. Il constitue une base solide pour des analyses plus avancées comme la priorisation des réclamations, l’analyse de sentiment, ou la détection des signaux de risque.

---

## Licence

À compléter selon le besoin du projet ou de l’utilisateur.
