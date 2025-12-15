"""
Evaluation Script für GPT-OSS-20B Model
Detaillierte Analyse und Metriken auf Test-Daten
"""
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    hamming_loss
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from data_loader import DuckDBDataLoader
from inference import DebateClassifier


class ModelEvaluator:
    """Evaluiert trainiertes Model auf Test-Daten"""
    
    def __init__(self, checkpoint_path: str, config_path: str):
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        
        # Config laden
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Classifier initialisieren
        self.classifier = DebateClassifier(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            threshold=0.5
        )
        
        # Data Loader
        self.data_loader = DuckDBDataLoader(config_path)
        
    def evaluate_on_test_set(self) -> Dict:
        """Evaluiert Model auf kompletten Test-Daten"""
        print("Lade Test-Daten...")
        _, test_df = self.data_loader.load_data()
        
        # Label-Spalten identifizieren
        label_cols = self.data_loader.get_label_columns(test_df)
        
        # Texte und Labels extrahieren
        texts = test_df[self.data_loader.text_column].tolist()
        true_labels = test_df[label_cols].values
        
        print(f"Evaluiere auf {len(texts)} Test-Samples...")
        
        # Predictions
        results = self.classifier.predict_batch(texts, batch_size=8)
        pred_labels = np.array([r['predictions'] for r in results])
        pred_probs = np.array([r['probabilities'] for r in results])
        
        # Metriken berechnen
        metrics = self._compute_metrics(true_labels, pred_labels, pred_probs, label_cols)
        
        return metrics, true_labels, pred_labels, pred_probs, label_cols
    
    def _compute_metrics(self, y_true, y_pred, y_prob, label_names) -> Dict:
        """Berechnet umfassende Metriken"""
        metrics = {}
        
        # Globale Metriken
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['hamming_loss'] = hamming_loss(y_true, y_pred)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Per-Label Metriken
        per_label_metrics = {}
        for i, label in enumerate(label_names):
            per_label_metrics[label] = {
                'f1': f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
                'precision': precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
                'recall': recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
                'support': int(y_true[:, i].sum())
            }
        
        metrics['per_label'] = per_label_metrics
        
        return metrics
    
    def print_metrics(self, metrics: Dict):
        """Druckt Metriken formatiert"""
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        
        print("\nGlobale Metriken:")
        print(f"  Accuracy:         {metrics['accuracy']:.4f}")
        print(f"  Hamming Loss:     {metrics['hamming_loss']:.4f}")
        print(f"  F1 (Macro):       {metrics['f1_macro']:.4f}")
        print(f"  F1 (Micro):       {metrics['f1_micro']:.4f}")
        print(f"  F1 (Weighted):    {metrics['f1_weighted']:.4f}")
        print(f"  Precision (Macro): {metrics['precision_macro']:.4f}")
        print(f"  Recall (Macro):   {metrics['recall_macro']:.4f}")
        
        print("\nPer-Label Metriken:")
        print(f"{'Label':<30} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Support':>8}")
        print("-" * 70)
        for label, m in metrics['per_label'].items():
            print(f"{label:<30} {m['f1']:>8.4f} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['support']:>8}")
    
    def save_metrics(self, metrics: Dict, output_path: str):
        """Speichert Metriken als JSON"""
        import json
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetriken gespeichert: {output_path}")
    
    def plot_confusion_matrices(self, y_true, y_pred, label_names, output_dir: Path):
        """Erstellt Confusion Matrices für jedes Label"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        n_labels = len(label_names)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, label in enumerate(label_names):
            if i >= len(axes):
                break
                
            cm = confusion_matrix(y_true[:, i], y_pred[:, i])
            
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                ax=axes[i],
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive']
            )
            axes[i].set_title(f'{label}')
            axes[i].set_ylabel('True')
            axes[i].set_xlabel('Predicted')
        
        # Leere Subplots ausblenden
        for i in range(n_labels, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        output_path = output_dir / "confusion_matrices.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Confusion Matrices gespeichert: {output_path}")
        plt.close()
    
    def plot_label_distribution(self, y_true, y_pred, label_names, output_dir: Path):
        """Vergleicht True vs. Predicted Label-Verteilung"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        true_counts = y_true.sum(axis=0)
        pred_counts = y_pred.sum(axis=0)
        
        x = np.arange(len(label_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, true_counts, width, label='True', alpha=0.8)
        ax.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8)
        
        ax.set_xlabel('Labels')
        ax.set_ylabel('Count')
        ax.set_title('Label Distribution: True vs. Predicted')
        ax.set_xticks(x)
        ax.set_xticklabels(label_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = output_dir / "label_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Label Distribution gespeichert: {output_path}")
        plt.close()


def main():
    """Haupt-Evaluation"""
    
    # Pfade
    checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "final_model"
    config_path = Path(__file__).parent.parent / "config" / "training_config.yaml"
    output_dir = Path(__file__).parent.parent / "evaluation_results"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluator initialisieren
    evaluator = ModelEvaluator(
        checkpoint_path=str(checkpoint_path),
        config_path=str(config_path)
    )
    
    # Evaluation durchführen
    metrics, y_true, y_pred, y_prob, label_names = evaluator.evaluate_on_test_set()
    
    # Metriken ausgeben
    evaluator.print_metrics(metrics)
    
    # Metriken speichern
    evaluator.save_metrics(metrics, str(output_dir / "metrics.json"))
    
    # Visualisierungen
    print("\nErstelle Visualisierungen...")
    evaluator.plot_confusion_matrices(y_true, y_pred, label_names, output_dir)
    evaluator.plot_label_distribution(y_true, y_pred, label_names, output_dir)
    
    print("\n" + "="*70)
    print("Evaluation abgeschlossen!")
    print(f"Ergebnisse gespeichert in: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
