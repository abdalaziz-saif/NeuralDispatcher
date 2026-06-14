"""
 Build Customer Dataset 
"""
import torch
from torch.utils.data import Dataset
from utils.logger import get_logger
from utils.exceptions import DataLoadingError

logger = get_logger(__name__)

class MyDataset(Dataset):


  def __init__(self, texts, labels, tokenizer):
    self.texts = texts
    self.labels = labels
    self.tokenizer = tokenizer

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, index):
    try:
      text = self.texts[index]
      label = self.labels[index]

      encoding = self.tokenizer(
        text,
        truncation=True,
        max_length=512,
        return_tensors='pt',
      )

      item = {
        'input_ids': encoding['input_ids'].squeeze(0),
        'attention_mask': encoding['attention_mask'].squeeze(0),
        'labels': torch.tensor(label, dtype=torch.long),
      }

      return item
    
    except Exception as e:
      logger.critical(f"Failed to load dataset item At index {index}: {e}")
      raise DataLoadingError(f"Dataset item error At index {index}: {e}") from e