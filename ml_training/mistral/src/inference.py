"""
Inference Script für trainiertes GPT-OSS-20B Model
Verwendet das fine-getunte Model für Multi-Label-Klassifikation
"""
import torch
import yaml
from pathlib import Path
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import numpy as np


class DebateClassifier:
    """Classifier für Brexit Debate Texte"""
    
    def __init__(self, checkpoint_path: str, config_path: str = None, threshold: float = 0.5):
        """
        Args:
            checkpoint_path: Pfad zum trainierten Model
            config_path: Pfad zur Config-Datei (optional)
            threshold: Schwellwert für Klassifikation (0-1)
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.threshold = threshold
        
        # Config laden falls vorhanden
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = None
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Verwende Device: {self.device}")
        
        # Model und Tokenizer laden
        self._load_model()
        
    def _load_model(self):
        """Lädt Model und Tokenizer"""
        print(f"Lade Model aus {self.checkpoint_path}...")
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.checkpoint_path))
        
        # Model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.checkpoint_path),
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None
        )
        
        self.model.eval()
        
        # Label Namen aus Config oder Model
        if hasattr(self.model.config, 'id2label'):
            self.label_names = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        else:
            self.label_names = [f"label_{i}" for i in range(self.model.config.num_labels)]
        
        print(f"Model geladen mit {self.model.config.num_labels} Labels")
        print(f"Label Namen: {self.label_names}")
    
    def predict(self, text: str) -> Dict:
        """
        Klassifiziert einen einzelnen Text
        
        Args:
            text: Input-Text
            
        Returns:
            Dictionary mit Predictions, Probabilities und Labels
        """
        # Tokenisierung
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=2048,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Zu Device verschieben
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
        # Sigmoid für Wahrscheinlichkeiten
        probabilities = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Predictions mit Threshold
        predictions = (probabilities > self.threshold).astype(int)
        
        # Ergebnis zusammenstellen
        result = {
            'text': text,
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'predicted_labels': [self.label_names[i] for i, pred in enumerate(predictions) if pred == 1],
            'label_scores': {label: float(prob) for label, prob in zip(self.label_names, probabilities)}
        }
        
        return result
    
    def predict_batch(self, texts: List[str], batch_size: int = 8) -> List[Dict]:
        """
        Klassifiziert mehrere Texte
        
        Args:
            texts: Liste von Input-Texten
            batch_size: Batch-Größe für effiziente Verarbeitung
            
        Returns:
            Liste von Prediction Dictionaries
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenisierung
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                max_length=2048,
                padding='max_length',
                return_tensors='pt'
            )
            
            # Zu Device verschieben
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Sigmoid für Wahrscheinlichkeiten
            probabilities = torch.sigmoid(logits).cpu().numpy()
            
            # Predictions mit Threshold
            predictions = (probabilities > self.threshold).astype(int)
            
            # Ergebnisse für jeden Text im Batch
            for j, text in enumerate(batch_texts):
                result = {
                    'text': text,
                    'predictions': predictions[j].tolist(),
                    'probabilities': probabilities[j].tolist(),
                    'predicted_labels': [self.label_names[k] for k, pred in enumerate(predictions[j]) if pred == 1],
                    'label_scores': {label: float(prob) for label, prob in zip(self.label_names, probabilities[j])}
                }
                results.append(result)
        
        return results
    
    def print_prediction(self, result: Dict):
        """Druckt Prediction-Ergebnis formatiert"""
        print(f"\nText: {result['text'][:100]}..." if len(result['text']) > 100 else f"\nText: {result['text']}")
        print(f"\nPredicted Labels: {', '.join(result['predicted_labels']) if result['predicted_labels'] else 'Keine'}")
        print("\nLabel Scores:")
        for label, score in sorted(result['label_scores'].items(), key=lambda x: x[1], reverse=True):
            bar = '█' * int(score * 20)
            print(f"  {label:20s} [{bar:20s}] {score:.3f}")


def main():
    """Beispiel-Verwendung"""
    
    # Pfade konfigurieren
    checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "final_model"
    config_path = Path(__file__).parent.parent / "config" / "training_config.yaml"
    
    # Classifier initialisieren
    classifier = DebateClassifier(
        checkpoint_path=str(checkpoint_path),
        config_path=str(config_path) if config_path.exists() else None,
        threshold=0.5
    )
    
    # Beispiel-Texte
    example_texts = [
        "The Brexit referendum has fundamentally changed the relationship between the UK and the European Union.",
        "Immigration policy was one of the key topics in the Brexit debate.",
        "The economic impact of leaving the single market is still unclear."
    ]
    
    # Einzelne Prediction
    print("\n" + "="*70)
    print("Einzelne Prediction:")
    print("="*70)
    result = classifier.predict(example_texts[0])
    classifier.print_prediction(result)
    
    # Batch Prediction
    print("\n" + "="*70)
    print("Batch Prediction:")
    print("="*70)
    results = classifier.predict_batch(example_texts)
    for i, result in enumerate(results, 1):
        print(f"\n--- Text {i} ---")
        classifier.print_prediction(result)


if __name__ == "__main__":
    main()
