import json

from .predict import Prediction
from .datasets import DatasetLoader

FILE_PATH = "../data/raw/"
FILE_NAME = "export_us_01.csv"

class Service:
    def __init__(self):
        self.prediction = Prediction()

    def create_answer(self, payload):
        output = self.prediction.predict(payload)
        result = {
            'input': payload['data'],
            'output': output
        }
        return result
