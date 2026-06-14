"""
trainer Module 
    - build train_one_epoch To train the model for one epoch
    - evaluate function to evaluate the model on the test set
"""
from utils.logger import get_logger
from utils.exceptions import TrainingError
from torch.amp import GradScaler, autocast

import tqdm
import torch
logger = get_logger(__name__)

# Training for one epoch function 
def train_one_epoch(train_split,
                    model,
                    loss_function,
                    optimizer,
                    scheduler,  
                    device,
                    scaler ):
    
    model.train() 

    correct_pred, total, total_loss = 0, 0, 0
    logger.info("Start training epoch")
    try:
        for batch in tqdm.tqdm(train_split):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            with autocast('cuda'):
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels)

                logits = outputs.logits
                loss = loss_function(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct_pred += (preds == labels).sum().item()
            total += labels.size(0)

    except Exception as e:
        logger.critical(f"Error happened during training: {e}")
        raise TrainingError(f"Training error: {e}") from e

    average_loss = total_loss / len(train_split) 
    logger.debug(f"Epoch training done Loss : {average_loss:.4f} - accuracy is : {correct_pred/total :.4f}")
    return average_loss, correct_pred / total

# Evaluate Function
def evaluate(test_split, model, loss_function, device):

    model.eval()

    correct, total_loss, total = 0, 0, 0
    all_preds, all_labels = [], []
    try:
        with torch.no_grad():
            for batch in test_split:

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels)
                logits = outputs.logits

                total_loss += loss_function(logits, labels).item()

                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    except Exception as e:
        logger.critical(f"Error happend during evaluation: {e}")
        raise TrainingError(f"Evaluation error: {e}") from e

    average_loss = total_loss / len(test_split) 
    return average_loss, correct / total , all_preds, all_labels