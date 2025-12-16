import torch


class Device:
    def __init__(self, model_path='model_0.pkl'):
        self.device = None

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def get_device(self):
        return self.device