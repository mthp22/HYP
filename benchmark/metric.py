import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

class MetricsCalculator:
    def __init__(self):
        self.true_labels = []
        self.pred_labels = []
        self.confidence = []
        self.execution_times = []

    def add_result(self, true_label, pred_label, confidence, exec_time):
        self.true_labels.append(true_label)
        self.pred_labels.append(pred_label)
        self.confidence.append(confidence)
        self.execution_times.append(exec_time)

    def calculate_metrics(self):
        if not self.true_labels:
            return {}
        
        tn, fp, fn, tp = confusion_matrix(self.true_labels, self.pred_labels).ravel()
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.true_labels,
            self.pred_labels,
            average='binary'
            )
        
        return {
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'avg_confidence': np.mean(self.confidences),
            'avg_execution_time': np.mean(self.execution_times),
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }

