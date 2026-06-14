"""
model Module 

    - Load Tokenizer & Model 
    - Fine_Tune the model 
    - Save & Load Model Functions  
    
"""


from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config import Config
from utils.logger import get_logger
from utils.exceptions import ModelLoadingError, ModelSavingError
import os ,json

logger = get_logger(__name__)

# Load Tokenizer
def load_tokenizer():
    try:
        return AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    except Exception as e:
        logger.critical(f"Failed to load tokenizer: {e}")
        raise ModelLoadingError(f"Failed to load tokenizer: {e}") from e

# Load Model
def load_model():
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            Config.MODEL_NAME,
            num_labels=getattr(Config, 'NUM_LABELS', None)
        )
        return fine_tune(model)
    except Exception as e:
        logger.critical(f"Failed to load model: {e}")
        raise ModelLoadingError(f"Failed to load model: {e}") from e

# Fine-tune the model 
def fine_tune(model , layers_to_train: int = None ):
   
    if layers_to_train == None :
        layers_to_train = Config.LAYERS_TO_TRAIN
    
    for param in model.parameters():
        param.requires_grad = False 

    
    if hasattr(model, 'roberta'):
        transformer_layers = model.roberta.encoder.layer
    elif hasattr(model, 'distilbert'):
        transformer_layers = model.distilbert.transformer.layer
    else:
        raise ValueError("Unsupported model type")
    
    # Unfreeze the last n layers
    for i in range (layers_to_train):
        
        for param in transformer_layers[-(i+1)].parameters():
            param.requires_grad = True 
    
    # Ensure the classification head is trainable
    if hasattr(model, 'classifier'):
        for param in model.classifier.parameters():
            param.requires_grad = True

    return model 


# save model and tokenizer 
def save_model(model, tokenizer, label_map):
    try:
        logger.info(f"Saving The model in path {Config.SAVE_PATH}")
        os.makedirs(Config.SAVE_PATH , exist_ok = True)
        model.save_pretrained(Config.SAVE_PATH)
        tokenizer.save_pretrained(Config.SAVE_PATH)
        with open (f"{Config.SAVE_PATH}/label_map.json" , 'w') as f :
            json.dump(label_map, f)
        logger.info(f"Saved to {Config.SAVE_PATH}")

    except Exception as e:
        logger.critical(f"Failed to save model: {e}")
        raise ModelSavingError(f"Failed to save model: {e}") from e


# Load Model and Tokenizer from saved path
def load_saved_model():
    try:
        logger.info(f"Loading The model & Toknizer From path {Config.SAVE_PATH}")
        model = AutoModelForSequenceClassification.from_pretrained(Config.SAVE_PATH)
        tokenizer = AutoTokenizer.from_pretrained(Config.SAVE_PATH)
        with open (f"{Config.SAVE_PATH}/label_map.json" ,'r') as f :
            label_map = json.load(f)
        label_map = {int(k):v for k,v in label_map.items()} # json saves the keys as str
        return model, tokenizer, label_map
    except Exception as e:
        logger.critical(f"Failed to load saved model: {e}")
        raise ModelLoadingError(f"Failed to load saved model: {e}") from e