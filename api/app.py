"""
App module 
    FastApi App
    Pydantic For Request/Respond validation 
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from model.model import load_saved_model
from utils.logger import get_logger
from inference.dispatcher import dispatch 

app = FastAPI()

logger = get_logger(__name__)

# load model and tonkenizer and label map 
model, tokenizer, label_map = load_saved_model()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
logger.info(f"loading model to {device}")

class DispatchRequest(BaseModel):
    text : str 

class DispatchRespond(BaseModel) :
    text : str 
    label : str

@app.get("/")
def root ():
    return {"message" : "hello and welocome to Neural Dispatcher  "}

@app.post("/dispatch", response_model= DispatchRespond) 
def inference(request: DispatchRequest):
    if not request.text.strip():
        raise HTTPException(status_code= 400 , detail = "Can't Procces empty string")
    
    try :
        prediction = dispatch(request.text, model, tokenizer, label_map, device)
        logger.info(f"Prediction for {request.text} is ->  {prediction}")
        return {'text':request.text , 'label':prediction} 
    except Exception as e :
        logger.critical(f"The inference Faild  {e}")
        raise HTTPException(state_code = 500 , detail = "inference Errore")


