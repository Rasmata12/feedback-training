import pandas as pd

df = pd.read_csv("data/base_ikan.csv")

avant = len(df)
df = df.drop_duplicates(subset=["texte"], keep="first")
apres = len(df)

df.to_csv("data/base_ikan.csv", index=False)
print(f"{avant - apres} doublon(s) supprimé(s). {apres} lignes restantes.")
