from .predict import Prediction
from .tags_gen import TagsGen
from .device import Device

class Service:
    def __init__(self):
        self.device = Device()
        self.prediction = Prediction(self.device)
        self.tags_gen = TagsGen()

    def create_answer(self, payload):
        tags = self.tags_gen.generate_client_tags(payload['data'])
        output = self.prediction.predict(tags)
        result = {
            'input': payload['data'],
            'output': output
        }
        return result
