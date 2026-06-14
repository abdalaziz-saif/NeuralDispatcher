import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from utils.logger import get_logger
from utils.exceptions import MetricsError

logger = get_logger(__name__)

def print_report(all_labels, all_preds, class_names):
    try:
        print("\n Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names))
    except Exception as e:
        logger.critical(f"Failed to print classification report: {e}")
        raise MetricsError(f"Classification report error: {e}") from e

def plot_confusion_matrix(all_labels, all_preds, class_names, save_path=None):
    try:
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d',
                    xticklabels=class_names,
                    yticklabels=class_names,
                    cmap='Blues')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('AI Request Dispatcher — Confusion Matrix', fontsize=14)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    except Exception as e:
        logger.critical(f"Failed to plot confusion matrix: {e}")
        raise MetricsError(f"Confusion matrix plotting error: {e}") from e