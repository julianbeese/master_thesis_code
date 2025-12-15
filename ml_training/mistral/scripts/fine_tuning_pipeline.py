#!/usr/bin/env python3
"""
Fine-Tuning Pipeline für Frame-Classification
Unterstützt verschiedene LLM-Ansätze (OpenAI, Hugging Face, etc.)
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import subprocess
import sys

class FineTuningPipeline:
    def __init__(self, training_data_dir: str, model_output_dir: str):
        self.training_data_dir = Path(training_data_dir)
        self.model_output_dir = Path(model_output_dir)
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        
    def check_dependencies(self, approach: str):
        """Prüft ob alle benötigten Dependencies installiert sind"""
        print(f"Prüfe Dependencies für {approach}...")
        
        if approach == "openai":
            try:
                import openai
                print("✓ OpenAI Python SDK gefunden")
            except ImportError:
                print("✗ OpenAI Python SDK nicht gefunden")
                print("Installiere mit: pip install openai")
                return False
                
        elif approach == "huggingface":
            try:
                import transformers
                import torch
                print("✓ Transformers und PyTorch gefunden")
            except ImportError:
                print("✗ Transformers oder PyTorch nicht gefunden")
                print("Installiere mit: pip install transformers torch")
                return False
                
        elif approach == "llama_cpp":
            try:
                import llama_cpp
                print("✓ llama-cpp-python gefunden")
            except ImportError:
                print("✗ llama-cpp-python nicht gefunden")
                print("Installiere mit: pip install llama-cpp-python")
                return False
        
        return True
    
    def prepare_openai_data(self) -> str:
        """Bereitet Daten für OpenAI Fine-Tuning vor"""
        print("Bereite OpenAI Fine-Tuning Daten vor...")
        
        # Lade Training-Daten
        train_file = self.training_data_dir / "train_alpaca.jsonl"
        if not train_file.exists():
            print(f"✗ Training-Datei {train_file} nicht gefunden!")
            return None
        
        # Konvertiere zu OpenAI-Format
        openai_data = []
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                openai_example = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "Du bist ein Experte für politische Diskursanalyse. Klassifiziere politische Reden in Frames: Human Impact, Powerlessness, Economic, Moral Value, Conflict, Other."
                        },
                        {
                            "role": "user", 
                            "content": data["input"]
                        },
                        {
                            "role": "assistant",
                            "content": data["output"]
                        }
                    ]
                }
                openai_data.append(openai_example)
        
        # Speichere OpenAI-Format
        output_file = self.model_output_dir / "openai_training_data.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in openai_data:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        print(f"✓ OpenAI-Daten gespeichert: {output_file}")
        return str(output_file)
    
    def run_openai_fine_tuning(self, training_file: str, api_key: str = None):
        """Führt OpenAI Fine-Tuning durch"""
        print("Starte OpenAI Fine-Tuning...")
        
        if not api_key:
            print("Warnung: Kein API-Key angegeben. Stelle sicher, dass OPENAI_API_KEY gesetzt ist.")
        
        # OpenAI CLI Kommando
        cmd = [
            "openai", "api", "fine_tuning.jobs.create",
            "-t", training_file,
            "-m", "gpt-3.5-turbo",
            "--suffix", "frame-classification"
        ]
        
        print(f"Führe aus: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("✓ Fine-Tuning Job erstellt")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"✗ Fehler beim Fine-Tuning: {e}")
            print(f"Stderr: {e.stderr}")
        except FileNotFoundError:
            print("✗ OpenAI CLI nicht gefunden. Installiere mit: pip install openai")
    
    def prepare_huggingface_data(self) -> Dict[str, str]:
        """Bereitet Daten für Hugging Face Fine-Tuning vor"""
        print("Bereite Hugging Face Fine-Tuning Daten vor...")
        
        # Lade Training-Daten (CSV Format)
        train_file = self.training_data_dir / "training_data_export.csv"
        test_file = self.training_data_dir / "test_data_export.csv"
        
        if not train_file.exists() or not test_file.exists():
            print("✗ Training/Test-Dateien nicht gefunden!")
            print(f"Erwartet: {train_file} und {test_file}")
            return None
        
        # Erstelle Dataset-Verzeichnis
        dataset_dir = self.model_output_dir / "hf_dataset"
        dataset_dir.mkdir(exist_ok=True)
        
        # Kopiere Dateien
        import shutil
        shutil.copy2(train_file, dataset_dir / "train.csv")
        shutil.copy2(test_file, dataset_dir / "test.csv")
        
        print(f"✓ Hugging Face Daten vorbereitet: {dataset_dir}")
        return {"dataset_dir": str(dataset_dir)}
    
    def create_huggingface_script(self, dataset_dir: str, model_name: str = "distilbert-base-uncased"):
        """Erstellt Hugging Face Fine-Tuning Script"""
        script_content = f'''#!/usr/bin/env python3
"""
Hugging Face Fine-Tuning Script für Frame-Classification
"""

import json
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Konfiguration
MODEL_NAME = "{model_name}"
DATASET_DIR = "{dataset_dir}"
OUTPUT_DIR = "frame_classification_model"
FRAME_LABELS = ["Human Impact", "Powerlessness", "Economic", "Moral Value", "Conflict", "Other"]

def load_data(file_path):
    """Lädt CSV-Daten"""
    import pandas as pd
    df = pd.read_csv(file_path)
    return df.to_dict('records')

def tokenize_function(examples):
    """Tokenisiert Texte"""
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)

def compute_metrics(eval_pred):
    """Berechnet Metriken"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)
    
    return {{
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }}

# Lade Daten
print("Lade Daten...")
train_data = load_data(f"{{DATASET_DIR}}/train.csv")
test_data = load_data(f"{{DATASET_DIR}}/test.csv")

# Erstelle Label-Mapping
label2id = {{label: i for i, label in enumerate(FRAME_LABELS)}}
id2label = {{i: label for label, i in label2id.items()}}

# Konvertiere Labels
for item in train_data + test_data:
    item["label"] = label2id[item["label"]]

# Erstelle Datasets
train_dataset = Dataset.from_list(train_data)
test_dataset = Dataset.from_list(test_data)

# Lade Tokenizer und Modell
print("Lade Modell und Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=len(FRAME_LABELS),
    id2label=id2label,
    label2id=label2id
)

# Tokenisiere Daten
print("Tokenisiere Daten...")
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
)

# Data Collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Training
print("Starte Training...")
trainer.train()

# Evaluation
print("Evaluierung...")
results = trainer.evaluate()
print(f"Ergebnisse: {{results}}")

# Speichere Modell
print("Speichere Modell...")
trainer.save_model()
tokenizer.save_pretrained(OUTPUT_DIR)

print("✓ Fine-Tuning abgeschlossen!")
'''
        
        script_path = self.model_output_dir / "hf_fine_tuning.py"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        print(f"✓ Hugging Face Script erstellt: {script_path}")
        return str(script_path)
    
    def create_evaluation_script(self):
        """Erstellt Evaluation-Script für das trainierte Modell"""
        eval_script = '''#!/usr/bin/env python3
"""
Evaluation Script für Frame-Classification Modell
"""

import json
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def load_model(model_path: str):
    """Lädt trainiertes Modell"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

def predict_frame(text: str, tokenizer, model):
    """Klassifiziert Text"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class_id = predictions.argmax().item()
        confidence = predictions.max().item()
    
    # Hole Label-Namen
    id2label = model.config.id2label
    predicted_label = id2label[predicted_class_id]
    
    return predicted_label, confidence

def main():
    parser = argparse.ArgumentParser(description='Evaluierung des Frame-Classification Modells')
    parser.add_argument('--model-path', required=True, help='Pfad zum trainierten Modell')
    parser.add_argument('--test-data', required=True, help='Pfad zu Test-Daten')
    parser.add_argument('--output', help='Ausgabedatei für Ergebnisse')
    
    args = parser.parse_args()
    
    # Lade Modell
    print("Lade Modell...")
    tokenizer, model = load_model(args.model_path)
    
    # Lade Test-Daten
    print("Lade Test-Daten...")
    test_data = []
    with open(args.test_data, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))
    
    # Evaluation
    print("Führe Evaluation durch...")
    correct = 0
    total = len(test_data)
    results = []
    
    for item in test_data:
        predicted_label, confidence = predict_frame(item['text'], tokenizer, model)
        actual_label = item['label']
        
        is_correct = predicted_label == actual_label
        if is_correct:
            correct += 1
        
        results.append({
            'text': item['text'][:100] + '...',  # Kürze für Ausgabe
            'actual': actual_label,
            'predicted': predicted_label,
            'confidence': confidence,
            'correct': is_correct
        })
    
    # Berechne Metriken
    accuracy = correct / total
    print(f"\\nErgebnisse:")
    print(f"Accuracy: {accuracy:.3f} ({correct}/{total})")
    
    # Zeige Beispiele
    print(f"\\nBeispiel-Vorhersagen:")
    for i, result in enumerate(results[:5]):
        status = "✓" if result['correct'] else "✗"
        print(f"{status} {result['actual']} → {result['predicted']} ({result['confidence']:.3f})")
    
    # Speichere Ergebnisse
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump({
                'accuracy': accuracy,
                'total_samples': total,
                'correct_predictions': correct,
                'results': results
            }, f, ensure_ascii=False, indent=2)
        print(f"\\nErgebnisse gespeichert: {args.output}")

if __name__ == "__main__":
    main()
'''
        
        script_path = self.model_output_dir / "evaluate_model.py"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(eval_script)
        
        print(f"✓ Evaluation Script erstellt: {script_path}")
        return str(script_path)
    
    def run_pipeline(self, approach: str, **kwargs):
        """Führt die komplette Fine-Tuning Pipeline aus"""
        print(f"Starte Fine-Tuning Pipeline für {approach}...")
        
        # Prüfe Dependencies
        if not self.check_dependencies(approach):
            return False
        
        if approach == "openai":
            # OpenAI Fine-Tuning
            training_file = self.prepare_openai_data()
            if training_file:
                self.run_openai_fine_tuning(training_file, kwargs.get('api_key'))
            
        elif approach == "huggingface":
            # Hugging Face Fine-Tuning
            dataset_info = self.prepare_huggingface_data()
            if dataset_info:
                script_path = self.create_huggingface_script(
                    dataset_info['dataset_dir'],
                    kwargs.get('model_name', 'distilbert-base-uncased')
                )
                print(f"\\nFühre Hugging Face Fine-Tuning aus:")
                print(f"python {script_path}")
            
        elif approach == "llama_cpp":
            print("Llama.cpp Fine-Tuning noch nicht implementiert")
            print("Verwende stattdessen Hugging Face oder OpenAI")
            
        else:
            print(f"Unbekannter Ansatz: {approach}")
            return False
        
        # Erstelle Evaluation Script
        self.create_evaluation_script()
        
        print(f"\\n✓ Pipeline für {approach} vorbereitet!")
        return True

def main():
    parser = argparse.ArgumentParser(description='Fine-Tuning Pipeline für Frame-Classification')
    parser.add_argument('--training-data', required=True, help='Verzeichnis mit Training-Daten')
    parser.add_argument('--output-dir', required=True, help='Ausgabeverzeichnis für Modelle')
    parser.add_argument('--approach', choices=['openai', 'huggingface', 'llama_cpp'], 
                       default='huggingface', help='Fine-Tuning Ansatz')
    parser.add_argument('--model-name', default='distilbert-base-uncased', 
                       help='Base-Modell für Hugging Face')
    parser.add_argument('--api-key', help='API-Key für OpenAI')
    
    args = parser.parse_args()
    
    # Prüfe Training-Daten
    if not Path(args.training_data).exists():
        print(f"✗ Training-Daten Verzeichnis {args.training_data} nicht gefunden!")
        return
    
    # Erstelle Pipeline
    pipeline = FineTuningPipeline(args.training_data, args.output_dir)
    
    # Führe Pipeline aus
    success = pipeline.run_pipeline(
        args.approach,
        model_name=args.model_name,
        api_key=args.api_key
    )
    
    if success:
        print(f"\\n✓ Fine-Tuning Pipeline erfolgreich!")
        print(f"\\nNächste Schritte:")
        print(f"1. Führe das generierte Script aus")
        print(f"2. Teste das trainierte Modell")
        print(f"3. Verwende das Modell für neue Klassifikationen")
    else:
        print(f"\\n✗ Pipeline fehlgeschlagen!")

if __name__ == "__main__":
    main()
