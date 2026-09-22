# Feedback Type Classification

Projet de machine learning conçu pour classifier automatiquement des retours clients en français selon leur nature : avis, réclamation, suggestion ou expérience utilisateur.

Ce projet a été développé dans une logique de portfolio IA/data science : démontrer la capacité à transformer un problème métier en pipeline de données, entraîner un modèle NLP, puis évaluer la qualité des prédictions sur un jeu de test réel.

---

## 🎯 Problème métier

Les entreprises reçoivent des milliers de commentaires clients, parfois très variés et peu structurés. Il est difficile de les classer rapidement et de comprendre s’ils correspondent à :

- une simple opinion,
- une réclamation,
- une demande d’amélioration,
- ou une expérience utilisateur positive/négative.

L’objectif de ce projet est de fournir une solution automatisée pour organiser et analyser ce type de feedback afin de mieux prioriser les actions à mener.

---

## 🧠 Approche technique

J’ai utilisé un modèle de transformer multilingue :

- XLM-RoBERTa
- fine-tuning sur un dataset de feedback multiclasse / multi-label
- classification binaire par label avec activation sigmoid
- seuil de décision fixé à 0.5

Le modèle est capable de détecter plusieurs catégories potentielles dans un même commentaire.

---

## 📊 Dataset

Le projet exploite des données de feedback utilisateur stockées dans le dossier `data/`.

Les fichiers clés sont :

- `data/base_ikan.csv` : base consolidée
- `data/train.csv` : données d’entraînement
- `data/val.csv` : données de validation
- `data/test.csv` : données de test
- `data/apia2022/` : données sources

Les labels utilisés sont :

- `AVIS`
- `RECLAMATION`
- `SUGGESTION`
- `EXPERIENCE`

---

## ⚙️ Pipeline du projet

### 1. Préparation des données
`01_prepare_data.py`

- charge les fichiers CSV sources,
- harmonise les colonnes,
- supprime les doublons et valeurs vides,
- nettoie les textes,
- sauvegarde la base finale.

### 2. Split train/validation/test
`02_split_data.py`

- répartit les données pour évaluer correctement le modèle,
- prépare les jeux d’entraînement et de test.

### 3. Entraînement du modèle
`03_train.py`

- charge `xlm-roberta-base`,
- configure le modèle pour la classification multi-label,
- entraîne le modèle sur 4 epochs,
- sauvegarde la version finale dans `model_feedback_type_final`.

### 4. Évaluation
`04_evaluate.py`

- charge le modèle entraîné,
- passe les données de test,
- produit un rapport de classification complet.

### 5. Analyse avancée
Le projet comporte aussi des modules pour aller au-delà du simple classement :

- `05_analyse_complete.py` : analyse complète du feedback
- `06_add_feedback.py` : ajout de nouvelles observations
- `08_discordance.py` : détection de discordance entre sentiment et type
- `09_criticite.py` : calcul de criticité
- `10_analyse_finale.py` : synthèse finale
- `13_dedupe.py` : suppression des doublons

---

## 📈 Résultats obtenus

Les résultats ci-dessous ont été mesurés sur le jeu de test réel du projet.

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

### Analyse des performances

- `RECLAMATION` est la catégorie la mieux détectée, avec un F1-score de `0.92`.
- `AVIS` est également très solide avec un F1-score de `0.88`.
- `SUGGESTION` reste bien performante à `0.87`.
- `EXPERIENCE` est la catégorie la plus difficile, avec `0.78`, probablement en raison du caractère plus subjectif et varié des commentaires.

Le score macro global de `0.86` montre une qualité de classification solide et exploitable pour un projet de type NLP appliqué.

---

## 🏆 Ce que ce projet montre

Ce projet illustre plusieurs compétences clés en data science et IA :

- préparation et nettoyage de données textuelles,
- transformation d’un besoin métier en problème ML,
- utilisation de modèles de langage avancés,
- fine-tuning d’un modèle de NLP,
- évaluation rigoureuse avec des métriques de classification,
- structuration d’un workflow de projet reproductible.

---

## 🚀 Réplication rapide

### Prérequis

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

## 📁 Structure du dépôt

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

> Les poids du modèle ne sont pas inclus dans le dépôt Git pour garder le projet léger. Il suffit de relancer l’entraînement pour reconstruire le modèle localement.

---

## 📌 Conclusion

Ce projet représente une application concrète de l’IA en traitement du langage naturel : automatiser la compréhension des feedback clients pour mieux les organiser, les prioriser et les exploiter.

C’est une base solide pour des extensions comme la détection de sentiment, l’analyse de risque, la priorisation des réclamations ou l’intégration dans un outil d’exploitation client.

---

## Licence

Projet personnel / portfolio. À adapter selon le contexte d’utilisation.
