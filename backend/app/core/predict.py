from time import sleep

import joblib
import torch
from transformers import AutoTokenizer


class Prediction:
    def __init__(self, model_path='model_0.pkl'):
        self.model = None
        self.device = None

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        try:
            self.model = joblib.load(model_path, mmap_mode="r")
        except Exception as e:
            print(f'Model at "{model_path}" load failure reason: {e}')
        except FileNotFoundError:
            print(f'Model at "{model_path}" not found')

        if self.model is not None:
            self.model.to(self.device)
            model_name = "t5-small"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def predict(self, input):
        if self.model is None:
            return "No answer"

        result_prediction = self.generate_tags(
            input,
            self.model,
            self.tokenizer)

        return result_prediction

    def generate_tags(self, query, model, tokenizer, max_length=512, num_beams=5):
        model.eval()
        inputs = tokenizer(query, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(
            model.device)

        with torch.no_grad():
            output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                decoder_start_token_id=tokenizer.pad_token_id  #  required for T5
            )
        return tokenizer.decode(output[0], skip_special_tokens=True)