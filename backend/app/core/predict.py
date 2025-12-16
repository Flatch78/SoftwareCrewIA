from time import sleep

import joblib
import torch
from transformers import AutoTokenizer


class Prediction:
    def __init__(self, device, model_path='model_0.pkl'):
        self.model = None
        self.device = device
        try:
            self.model = joblib.load(model_path, mmap_mode="r")
            # .to(self.device))
            # self.model = torch.load(model_path, map_location=self.device)
        except Exception as e:
            print(f'Model at "{model_path}" load failure reason: {e}')
        except FileNotFoundError:
            print(f'Model at "{model_path}" not found')

        if self.model is not None:
            # self.model.to(self.device)
            model_name = "t5-small"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def predict(self, input_predict):
        if self.model is None:
            return "No answer"

        input_desc = self.generate_input_description(input_predict)

        result_prediction = self.generate_description(
            input_desc,
            self.model,
            self.tokenizer)

        return result_prediction

    def generate_input_description(self, tags):
        return f"""
        Analyze the following tags relevant for this description
        Extract only the relevant business requirement and create a complete Agile User Story.
        The User Story MUST include:
        Title:
        User Story:
        Acceptance Criteria:
        Tags:
        {tags}
        User Story:
        """

    def generate_description(self, query, model, tokenizer, max_length=512, num_beams=5):
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
                decoder_start_token_id=tokenizer.pad_token_id  # required for T5
            )
        return tokenizer.decode(output[0], skip_special_tokens=True)
