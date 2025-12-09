import os
import pandas as pd

class DatasetLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load(self):
        print(f"📄 Chargement du dataset : {self.filepath}")

        try:
            if not os.path.exists(self.filepath):
                raise FileNotFoundError(f"Fichier non trouvé: {self.filepath}")
            df = pd.read_csv(self.filepath, low_memory=False, sep=";", encoding="utf-8")
            print(f"CSV file chargé avec succès. Nombre d'échantillons chargés: {df.shape[0]}")
        except FileNotFoundError as e:
            raise Exception(e)

        # Exemple générique : "input" → X, "target" → y
        # if "input" not in df.columns or "target" not in df.columns:
        #     raise ValueError("Le dataset doit contenir les colonnes 'input' et 'target'.")
        #
        # X = df["input"]
        # y = df["target"]

        # Clean data
        columns = ['Key', 'Created']
        df.drop(columns=columns, inplace=True)


        # récupération
        df["New_description"] = df[
            ["Description", "Unnamed: 7", "Unnamed: 8", "Unnamed: 9", "Unnamed: 10", "Unnamed: 11"]].apply(
            lambda x: " ".join(map(str, x.dropna())), axis=1)

        # Corpus étendu de textes en français (400+ phrases)
        corpus_francais = df['New_description']

        print(f"📚 Corpus créé : {len(corpus_francais)} phrases")
        print(f"\n📝 Exemples de phrases :")
        for i, phrase in enumerate(corpus_francais[:2], 1):
            print(f"  {i}. {phrase}")

        print("Dataset chargé ✔")
        print(f"Nombre d'échantillons : {len(df)}")

        return X, y
