import pandas as pd
import glob

fichiers = glob.glob("data/apia2022/*.csv")
dfs = []
for f in fichiers:
    d = pd.read_csv(f)
    d["app_source"] = f.split("/")[-1].replace(".csv", "")
    dfs.append(d)
df = pd.concat(dfs, ignore_index=True)

df = df.rename(columns={
    "data": "texte",
    "rating": "AVIS",
    "bug_report": "RECLAMATION",
    "feature_request": "SUGGESTION",
    "user_experience": "EXPERIENCE",
})

df = df.dropna(subset=["texte"])
df = df.drop_duplicates(subset=["texte"])
df["texte"] = df["texte"].astype(str).str.strip()

for cat in ["AVIS", "RECLAMATION", "SUGGESTION", "EXPERIENCE"]:
    df[cat] = df[cat].fillna(0).astype(int)

df["source"] = "dataset_apia2022"
df["valide_humain"] = 1

df = df[["texte", "AVIS", "RECLAMATION", "SUGGESTION", "EXPERIENCE", "source", "valide_humain"]]
df.to_csv("data/base_ikan.csv", index=False, encoding="utf-8")

print(f"Base créée : {len(df)} lignes -> data/base_ikan.csv")
print(df[["AVIS", "RECLAMATION", "SUGGESTION", "EXPERIENCE"]].sum())
