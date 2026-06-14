""" 
preprocessing 

    - Load_data from Kagglehub 
    - build label map 
    - build dataset using the custom dataset class
    - split dataset to train , test
    - build dataloader for train and test
    - use collator to handle padding
    - build class weight function to handle class imbalance
    - custom logging and exception handling 
"""






import numpy as np
import pandas as pd
import kagglehub
import transformers
import torch
from torch.utils.data import random_split, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from .dataset import MyDataset
from config import Config
from utils.logger import get_logger
from utils.exceptions import DataLoadingError

logger = get_logger(__name__)

# Load data 
def load_data():
    try : 
        logger.info("Data Loading ..")
        path = kagglehub.dataset_download(Config.DATASET_NAME)
        data = pd.read_csv(f"{path}/dolly_15k.csv")
        data = data.drop(columns=['context', 'response'])
        return data
    except Exception as e :
        logger.critical(f"Failed to load data {e}")
        raise DataLoadingError (f"Data Loading Errore{e}")

# build label mapping 
def build_label_map(data):
    try:
     
        category_mapping = {cat: i for i, cat in enumerate(sorted(data['category'].unique()))}
        label_map = {v: k for k, v in category_mapping.items()}
        return category_mapping, label_map
    
    except Exception as e:
        logger.critical(f"Failed to build label map: {e}")
        raise DataLoadingError(f"Label map construction error: {e}") from e

# build dataset 
def build_datasets(data: pd.DataFrame, tokenizer, category_mapping: dict[str, int]):
    try:
        
        logger.info(f"Building the dataset with batch size {Config.BATCH_SIZE}")
        texts = data['instruction'].tolist()
        labels = data['category'].map(category_mapping).tolist()

        full_data = MyDataset(texts, labels, tokenizer)

        train_size = int(0.80 * len(full_data))
        test_size = len(full_data) - train_size

        return random_split(full_data, [train_size, test_size])
    except Exception as e:
        logger.critical(f"Failed to build datasets: {e}")
        raise DataLoadingError(f"Dataset build  error: {e}") from e



#  build The DataLoader 
def create_dataloaders(train_data, val_data, tokenizer):
    try:
        collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)
        logger.info(f"Building the dataloader")
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            collate_fn=collator,
        )

        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collator,
        )

        return train_loader, val_loader
    except Exception as e:
        logger.critical(f"Failed to create dataloaders: {e}")
        raise DataLoadingError(f"Dataloader creation error: {e}") from e

# Class weight Function 

def get_class_weights(train_loader, device):
    try:
        all_labels = []
        for batch in train_loader:
            labels = batch.get('labels') if isinstance(batch, dict) else batch['labels']
            if isinstance(labels, torch.Tensor):
                all_labels.extend(labels.cpu().numpy().tolist())
            else:
                all_labels.extend(labels)

        weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(all_labels),
            y=all_labels,
        )
        return torch.tensor(weights, dtype=torch.float).to(device)
    except Exception as e:
        logger.critical(f"Failed to compute class weights: {e}")
        raise DataLoadingError(f"Class weight comp error: {e}") from e

