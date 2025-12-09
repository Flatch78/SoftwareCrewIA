import json

from .train import Trainer
from .datasets import DatasetLoader

FILE_PATH = "../data/raw/"
FILE_NAME = "export_us_01.csv"


def create_answer(payload):
    result = {
        'input': payload['data'],
        'output': 'Output'
    }

    return result


def training():
    # Charger le dataset
    file_path = FILE_PATH + FILE_NAME
    dataset = DatasetLoader(filepath=f"{file_path}")
    try:
        X, y = dataset.load()
    except Exception as e:
        print(e)
        return

    # Créer un trainer et lancer l'entraînement
    trainer = Trainer()
    model = trainer.train(X, y)

    # Sauvegarder le modèle
    trainer.save_model(model, save_path="models/model.pkl")

    print("🎉 Entraînement terminé et modèle sauvegardé !")
