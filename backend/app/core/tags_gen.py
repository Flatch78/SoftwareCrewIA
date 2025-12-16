import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class TagsGen:
    def __init__(self):
        self.device = None

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.configured = False
        if self.device is not None:
            self.configured = True
            model_name = "google/flan-t5-large"

            self.auto_tokenizer_gen = AutoTokenizer.from_pretrained(model_name)
            self.model_gen = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model_gen.to(self.device)

    def generate_client_tags(self, target, max_length=64):
        def format_prompt(desc, num_tags=8):
            return f"""
            Generate {num_tags} relevant tags for this description.
            Tags should be lowercase, comma-separated, and include technologies, frameworks, and project type.
            Project description: {desc}
            Tags:
            """
        if self.configured:
            prompt = format_prompt(target)

            inputs_gen = self.auto_tokenizer_gen(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                outputs_gen = self.model_gen.generate(
                    **inputs_gen,
                    max_length=max_length,
                    num_beams=4,
                    no_repeat_ngram_size=3,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    early_stopping=True,
                )

            tags = self.auto_tokenizer_gen.decode(outputs_gen[0], skip_special_tokens=True)

            # Clean up output
            tags = tags.strip()
            if not tags:
                return "web-app, software, development"  # Fallback

            return tags
        return "empty, web-app, software, development"