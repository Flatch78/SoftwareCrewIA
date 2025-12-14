from fastapi import FastAPI
from data.models import UserCaseRequest
from core.service import Service

app = FastAPI()
service = Service()

@app.get("/hello")
def hello():
    return {"message": "Hello from FastAPI Backend!"}

@app.get("/")
def root():
    return {"message": "Fast API in Python"}


@app.post("/create", status_code=201)
def create_answer(payload: UserCaseRequest):
    payload = payload.model_dump()
    return service.create_answer(payload)
