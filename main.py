from json import load

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from transformers import get_linear_schedule_with_warmup

from config import Config
from data.preprocessing import (create_dataloaders, load_data, build_label_map,
                                 build_datasets, get_class_weights)
from model.model import load_tokenizer, load_model, save_model , load_saved_model
from model.trainer import train_one_epoch, evaluate
from utils.metrics import print_report, plot_confusion_matrix
from inference.dispatcher import dispatch
from utils.logger import get_logger

logger = get_logger(__name__)

def main():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # data 
        # ___________________________________________________________

        data = load_data()
        logger.info(f"Data loaded, shape: {data.shape}")

        category_mapping, label_map = build_label_map(data)
        # build ordered class names by label index
        class_names = [label_map[i] for i in sorted(label_map.keys())]
        # tokenizer
        tokenizer = load_tokenizer()
        # split data
        train_split, val_split = build_datasets(data, tokenizer, category_mapping)
        logger.info("Data split finished")

        # create dataloader
        train_loader, val_loader = create_dataloaders(train_split, val_split, tokenizer)
        logger.info(f"Data loader done, train_loader length is : {len(train_loader)}")


        # MODEL 
        #_______________________________________________________________

        # model
        model = load_model()
        model.to(device)

        # get the class weight to address the class impalance 
        class_weights = get_class_weights(train_loader, device)

        #lossFunction 
        loss_function = nn.CrossEntropyLoss(weight=class_weights)

        # optimizer and scheduler 
        optimizer = optim.AdamW([
            {'params': model.roberta.parameters(),   'lr': Config.LEARNING_RATE},
            {'params': model.classifier.parameters(), 'lr': Config.CLASSIFIER_LR}
        ])
        total_steps = len(train_loader) * Config.EPOCHS
        scheduler   = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(Config.WARMUP_RATIO * total_steps),
            num_training_steps=total_steps
        )
        scaler = GradScaler('cuda')

        # training 
        for epoch in range(Config.EPOCHS):
            logger.info(f"Epoch {epoch+1}/{Config.EPOCHS}")

            train_loss, train_acc = train_one_epoch(
                train_loader, model, loss_function, optimizer, scheduler, device, scaler)

            val_loss, val_acc, all_preds, all_labels = evaluate(
                val_loader, model, loss_function, device)

            logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

            if epoch == Config.EPOCHS - 1:
                print_report(all_labels, all_preds, class_names)
                plot_confusion_matrix(all_labels, all_preds, class_names,
                                      save_path="./utils/confusion_matrix.png")
                

        # save model  
        save_model(model , tokenizer, label_map)

    except Exception as e : 
        logger.critical(f"An error occurred in the main function: {e}")
        raise e 
    
      
# Dispatcher demo
# def dispatch_demo (label_map , device ):
#         model, tokenizer = load_saved_model()
#         logger.info("--- Dispatcher Demo ---")
#         model.eval()
#         tests = [
#             "What is the capital of Sudan?",
#             "Write me a poem about Vikings",
#             "Summarize this article ",
#             "Brainstorm ideas for a project"]

#         for text in tests:
#             logger.info(f"request : {text}")
#             logger.info(f"Category : {dispatch(text, model, tokenizer, label_map, device)}")




if __name__ == '__main__':
    main()
  