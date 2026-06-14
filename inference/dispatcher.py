"""
dispatcher Module 

    - predict inputs 
"""

import torch 
from utils.logger import get_logger
from utils.exceptions import InferenceError

logger = get_logger(__name__)

def dispatch(text, model, tokenizer, label_map, device):
    try:
        logger.debug("Dispatching text")

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        logger.debug(f"Dispatching text: {text}")
        with torch.no_grad():
            logits = model(**inputs).logits

        pred = torch.argmax(logits, dim=1).item()
        label = label_map[pred]
        logger.debug(f"Predictions: {label}")
        return label

    except KeyError as e:
        logger.critical(f"Label mapping failed during inference: {e}")
        raise InferenceError(f"Inference label mapping error: {e}") from e
    except Exception as e:
        logger.critical(f"Error occurred during inference: {e}")
        raise InferenceError(f"Inference error: {e}") from e

