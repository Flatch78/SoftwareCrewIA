import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings

warnings.filterwarnings('ignore')

import re
import datetime

from typing import List
import pandas as pd
from datasets import (
    DatasetDict,
    Dataset
)
from peft import (
    LoraConfig,
    get_peft_model,
)
import torch
from transformers import (
    T5ForConditionalGeneration,
    Trainer,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
)
import joblib


def train():
    csv_path = "./data/raw/export_us_01.csv"
    model, trainer = load_dataset(csv_path)

    print("Début de l'entrainement.")

    trainer.train()

    print("Fin de l'entrainement.")

    x = datetime.datetime.now()
    x = x.strftime("%Y-%m-%d.%H:%M:%S")
    modelName = f"model_0_{x}.pkl"
    # modelName = f"model_0.pkl"

    print(f"• enregistrement du modèle {modelName}")
    joblib.dump(model, "models/" + modelName)
    print("• Fin de l'enregistrement' du modèle")


def load_dataset(csv_path):
    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Fichier non trouvé: {csv_path}")
        df = pd.read_csv(csv_path, low_memory=False, sep=";", encoding="utf-8")
        print(f"CSV file chargé avec succès. Nombre d'échantillons chargés: {df.shape[0]}")
    except FileNotFoundError as e:
        print(f"Chargement erreur : {e}")
        return

    print("Chargement du dataset fini.")

    # Clean data
    columns = ['Key', 'Created']
    df.drop(columns=columns, inplace=True)
    features = ['Issue Type', 'Summary', 'Description']
    contentX = df[features].copy()
    contentX.fillna("")

    print("Nettoyage du dataset fini.")

    def nettoyer_texte_description(text: str) -> str:
        text = text.strip()
        text = re.sub(r' +', ' ', text)  # espaces multiples
        text = re.sub(r'\n\s*\n+', '\n', text)  # supprime lignes vides multiples
        text = text.replace(r'\n\s*\n+', '')
        return text

    # Nettoie et normalise le texte
    def nettoyer_texte(texte):
        texte = re.sub(r"([.,!?'])", r" \1 ", texte)
        texte = re.sub(r"([-●'])", r" ", texte)
        return texte.strip()

    def extract_acceptance_criteria(text: str) -> List[str]:
        text = text.replace("●", "-")  # Remplace les bullets non standard par "-"

        # Trouver la section "Acceptance Criteria"
        match = re.search(r'Acceptance Criteria(.*)', text, re.DOTALL | re.IGNORECASE)
        if not match:
            return []

        ac_section = match.group(1).strip()

        # Découper selon les puces commençant par "-"
        items = re.split(r'-\s*', ac_section)
        items = [i.strip() for i in items if i.strip()]

        return items

    def safe_text(v):
        if isinstance(v, float):  # couvre NaN ou nombres
            return ""
        return str(v).strip()

    def preprocess_issueType(raw_text: str) -> str:
        return raw_text

    def preprocess_summary(raw_text: str) -> str:
        return raw_text

    def preprocess_description(raw_text: str):
        text = nettoyer_texte_description(raw_text)

        # Résumé auto : première phrase "As a business..."
        summary_match = re.search(r"As a .*?[\.\n]", text, re.IGNORECASE)
        summary = summary_match.group(0).strip() if summary_match else ""

        # Description : la partie avant les critères d'acceptation
        description = re.split(r'Acceptance Criteria', text, flags=re.IGNORECASE)[0]
        description = nettoyer_texte_description(description)

        # Critères d'acceptation
        acceptance_criteria = extract_acceptance_criteria(text)
        acceptance_criterias = '\n - '.join(acceptance_criteria)
        return {
            'content_summary': summary,
            'description': description,
            'acceptance_criteria': acceptance_criterias,
        }

    resp = []

    for i, phrase in contentX.iterrows():
        rawIssueType = safe_text(phrase['Issue Type'])
        if rawIssueType == "":
            rawIssueType = "Story"  # set default value
        rawSummary = safe_text(phrase['Summary'])
        if rawSummary == "":
            rawIssueType = "Empty"  # set default value
        rawDescription = safe_text(phrase['Description'])
        # Skip empty info
        if rawDescription != "":
            issueType = preprocess_issueType(rawIssueType)
            summary = preprocess_summary(rawSummary)
            description = preprocess_description(rawDescription)
            resp.append(
                {
                    "issue_type": nettoyer_texte(issueType),
                    "summary": nettoyer_texte(rawSummary),
                    "content_summary": nettoyer_texte(description['content_summary']),
                    "description": nettoyer_texte(description['description']),
                    "acceptance_criteria": nettoyer_texte(description['acceptance_criteria']),
                })

    print("Construction du dataset fini.")

    use_mps = False
    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_fp16 = True
        print(f"GPU NVIDIA détecté: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        use_mps = True
        use_fp16 = False
        print("GPU Apple Silicon (MPS) détecté")
    else:
        device = torch.device("cpu")
        use_mps = False
        use_fp16 = False
        print("CPU détecté")

    print("Définition du device fini.")

    model_name = "google/flan-t5-base"

    autoTokenizerGen = AutoTokenizer.from_pretrained(model_name)
    modelGen = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    modelGen.to(device)

    def generate_client_sentence(target, max_length=64):
        def format_prompt(target, num_tags=8):
            return f"""
            Generate {num_tags} relevant tags for this description.
            Tags should be lowercase, comma-separated, and include technologies, frameworks, and project type.

            Project description: {target}

            Tags:
            """

        prompt = format_prompt(target)

        inputs_gen = autoTokenizerGen(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            outputs_gen = modelGen.generate(
                **inputs_gen,
                max_length=max_length,
                num_beams=4,
                no_repeat_ngram_size=3,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                early_stopping=True,
            )

        tags = autoTokenizerGen.decode(outputs_gen[0], skip_special_tokens=True)

        # Clean up output
        tags = tags.strip()
        if not tags:
            return "web-app, software, development"  # Fallback

        return {
            "target": target,
            "tags": tags,
        }

    def generate_input_description(tags):
        return f"""
        Analyze the following tags relevant for this description
        Extract only the relevant business requirement and create a complete Agile User Story.
        Tags: {tags}
        User Story:
        """

    content_data = []
    for t in resp:
        data = generate_client_sentence(t['description'])
        data_input = generate_input_description(data['tags'])
        content_data.append({
            "input": data_input,
            "output": data['target'],
        })

    print("Generation du Dataset fini.")

    split = Dataset.from_list(content_data).train_test_split(test_size=0.1, seed=42)
    dataset_dict = DatasetDict({
        "train": split["train"],
        "validation": split["test"]
    })

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess(batch):
        inputs = batch["input"]
        targets = batch["output"]
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length").input_ids
        model_inputs["labels"] = labels
        return model_inputs

    tokenized = dataset_dict.map(preprocess, batched=True, remove_columns=dataset_dict["train"].column_names)
    tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    print("Construction du dictionnaire du Dataset fini.")

    model = T5ForConditionalGeneration.from_pretrained(model_name)
    lora_config = LoraConfig(
        r=16,  # defines the rank of the update matrices
        lora_alpha=32,  # scales the updates
        target_modules=["q", "k", "v", "o"],  # Adjust based on model architecture - attention projection modules
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM"  # sequence-to-sequence task
    )

    model = get_peft_model(model, lora_config)
    # Paramétres
    EPOCHS = 25  # 25
    LEARNING_RATE = 5e-5

    train_batch_size = 8 if use_mps else 4
    eval_batch_size = 8 if use_mps else 4

    training_args = TrainingArguments(
        output_dir="./target/t5_tag_generator",

        # Paramètres d'entraînement
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,

        # Paramètres essentiels selon le device
        fp16=use_fp16,  # True si GPU
        use_mps_device=use_mps,  # True si GPU mps détecté

        # Optimisation
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,

        logging_steps=10,
        eval_strategy="steps",
        eval_steps=500,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator
    )

    print("Préparation de la configuration pour l'entrainement du modèle fini.")

    return model, trainer
