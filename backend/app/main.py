from fastapi import FastAPI
from data.models import UserCaseRequest
from data.train import train
from core.service import Service

app = FastAPI()
service = Service()

@app.get("/train")
def train_model():
    print("Début de l'entraînement")
    train()
    print("Fin de l'entraînement")
    return {"message": "Modèle entraîné avec succès"}

@app.get("/")
def root():
    return {"message": "Fast API in Python"}


@app.post("/create", status_code=201)
def create_answer(payload: UserCaseRequest):
    payload = payload.model_dump()
    resp = service.create_answer(payload)
    print(f"payload {payload}\nResp {resp}\n")
    return resp
