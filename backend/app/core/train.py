import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class Trainer:
    def __init__(self):
        # Le pipeline ML de base : vectorisation + modèle
        self.pipeline = Pipeline([
            ("vectorizer", TfidfVectorizer()),
            ("classifier", LogisticRegression())
        ])

    def train(self, X, y):
        print("🔄 Entraînement du modèle...")
        self.pipeline.fit(X, y)
        return self.pipeline

    def save_model(self, model, save_path: str):
        joblib.dump(model, save_path)
        print(f"📁 Modèle sauvegardé dans : {save_path}")

    def load_model(self, save_path: str):
        print(f"📂 Chargement du modèle depuis : {save_path}")
        return joblib.load(save_path)
