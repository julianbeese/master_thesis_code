#!/usr/bin/env python3
"""
Generiert Training-Daten für Frame-Classification aus annotierten Chunks
Unterstützt verschiedene Formate für verschiedene LLM-Fine-Tuning-Ansätze
"""

import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import random
from collections import Counter

class TrainingDataGenerator:
    def __init__(self, annotations_file: str, output_dir: str):
        self.annotations_file = annotations_file
        self.output_dir = Path(output_dir)
        self.annotations = []
        self.frame_categories = [
            "Human Impact",
            "Powerlessness", 
            "Economic",
            "Moral Value",
            "Conflict",
            "Other"
        ]
        
    def load_annotations(self):
        """Lädt Annotationen aus JSON-Datei"""
        print(f"Lade Annotationen aus {self.annotations_file}...")
        
        with open(self.annotations_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.annotations = list(data['annotations'].values())
        
        print(f"✓ {len(self.annotations)} Annotationen geladen")
        
        # Zeige Verteilung
        frame_counts = Counter(ann['frame_label'] for ann in self.annotations)
        print("\nFrame-Verteilung:")
        for frame, count in frame_counts.most_common():
            percentage = (count / len(self.annotations) * 100)
            print(f"  {frame:15}: {count:4} ({percentage:5.1f}%)")
    
    def create_train_test_split(self, test_size: float = 0.2, random_seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
        """Erstellt Train/Test-Split mit stratifizierter Stichprobe"""
        print(f"Erstelle Train/Test-Split (Test: {test_size:.1%})...")
        
        train_data = []
        test_data = []
        
        # group by existing split
        for ann in self.annotations:
            dataset_split = ann.get('dataset_split', '').lower()
            
            if dataset_split == 'train':
                train_data.append(ann)
            elif dataset_split == 'test':
                test_data.append(ann)
            else:
                print(f"Warning: unknown dataset_split Value: {dataset_split}")
        
        # Zeige Verteilung pro Frame
        train_counts = Counter(ann['frame_label'] for ann in train_data)
        test_counts = Counter(ann['frame_label'] for ann in test_data)
        
        print("\nTrain-Set Verteilung:")
        for frame in self.frame_categories:
            count = train_counts.get(frame, 0)
            print(f"  {frame:15}: {count:3}")
        
        print("\nTest-Set Verteilung:")
        for frame in self.frame_categories:
            count = test_counts.get(frame, 0)
            print(f"  {frame:15}: {count:3}")
        
        print(f"\n✓ Train: {len(train_data)}, Test: {len(test_data)}")
        return train_data, test_data
    
    def generate_classification_format(self, data: List[Dict], format_type: str = "jsonl") -> List[str]:
        """Generiert Daten im Classification-Format"""
        if format_type == "jsonl":
            return self._generate_jsonl_format(data)
        elif format_type == "csv":
            return self._generate_csv_format(data)
        elif format_type == "alpaca":
            return self._generate_alpaca_format(data)
        else:
            raise ValueError(f"Unbekanntes Format: {format_type}")
    
    def _generate_jsonl_format(self, data: List[Dict]) -> List[str]:
        """Generiert JSONL-Format für Classification"""
        lines = []
        for ann in data:
            example = {
                "text": ann['chunk_text'],
                "label": ann['frame_label'],
                "metadata": {
                    "chunk_id": ann['chunk_id'],
                    "speaker": ann['speaker_name'],
                    "party": ann['speaker_party'],
                    "confidence": ann['confidence'],
                    "notes": ann['notes']
                }
            }
            lines.append(json.dumps(example, ensure_ascii=False))
        return lines
    
    def _generate_csv_format(self, data: List[Dict]) -> List[str]:
        """Generiert CSV-Format mit chunk_text und frame_label Spalten"""
        lines = ["chunk_text,frame_label,confidence,notes"]
        for ann in data:
            text = ann['chunk_text'].replace('"', '""')  # Escape quotes
            label = ann['frame_label']
            confidence = ann['confidence']
            notes = ann['notes'].replace('"', '""') if ann['notes'] else ""
            lines.append(f'"{text}","{label}",{confidence},"{notes}"')
        return lines
    
    def _generate_alpaca_format(self, data: List[Dict]) -> List[str]:
        """Generiert Alpaca-Format für Instruction-Following"""
        lines = []
        for ann in data:
            example = {
                "instruction": "Classify the following political speech text into one of these frames: Human Impact, Powerlessness, Economic, Moral Value, Conflict, or Other. Consider the main themes, emotional tone, and arguments presented.",
                "input": ann['chunk_text'],
                "output": ann['frame_label']
            }
            lines.append(json.dumps(example, ensure_ascii=False))
        return lines
    
    def generate_few_shot_examples(self, n_examples: int = 5) -> List[Dict]:
        """Generiert Few-Shot-Beispiele für jede Frame-Kategorie"""
        print(f"Generiere {n_examples} Few-Shot-Beispiele pro Kategorie...")
        
        examples = []
        for frame in self.frame_categories:
            # Finde beste Beispiele für diese Kategorie
            frame_annotations = [ann for ann in self.annotations if ann['frame_label'] == frame]
            
            if not frame_annotations:
                continue
            
            # Sortiere nach Confidence (höchste zuerst)
            frame_annotations.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Nimm die besten Beispiele
            selected = frame_annotations[:n_examples]
            examples.extend(selected)
        
        print(f"✓ {len(examples)} Few-Shot-Beispiele generiert")
        return examples
    
    def create_prompt_template(self) -> str:
        """Erstellt Prompt-Template für Frame-Classification"""
        template = """Du bist ein Experte für politische Diskursanalyse. Deine Aufgabe ist es, politische Reden in verschiedene Frames zu klassifizieren.

FRAME-KATEGORIEN:
- Human Impact: Fokus auf menschliche Auswirkungen, persönliche Geschichten, Betroffenheit
- Powerlessness: Gefühl der Ohnmacht, fehlende Kontrolle, Hilflosigkeit
- Economic: Wirtschaftliche Argumente, Kosten, Nutzen, Finanzen
- Moral Value: Ethische/moralische Argumente, Werte, Prinzipien
- Conflict: Konflikt, Opposition, Widerstand, Kampf
- Other: Sonstige Kategorien, die nicht in die obigen passen

BEISPIEL-ANNOTATIONEN:
{examples}

AUFGABE:
Klassifiziere den folgenden Text in eine der Frame-Kategorien. Gib nur den Frame-Namen als Antwort.

TEXT:
{text}

FRAME:"""
        return template
    
    def save_training_data(self, train_data: List[Dict], test_data: List[Dict]):
        """Speichert Training-Daten in verschiedenen Formaten"""
        print("Speichere Training-Daten...")
        
        # Erstelle Output-Verzeichnisse
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # JSONL-Format
        train_jsonl = self.generate_classification_format(train_data, "jsonl")
        test_jsonl = self.generate_classification_format(test_data, "jsonl")
        
        with open(self.output_dir / "train.jsonl", 'w', encoding='utf-8') as f:
            f.write('\n'.join(train_jsonl))
        
        with open(self.output_dir / "test.jsonl", 'w', encoding='utf-8') as f:
            f.write('\n'.join(test_jsonl))
        
        # CSV-Format (mit Dateinamen aus training_config.yaml)
        train_csv = self.generate_classification_format(train_data, "csv")
        test_csv = self.generate_classification_format(test_data, "csv")
        
        with open(self.output_dir / "training_data_export.csv", 'w', encoding='utf-8') as f:
            f.write('\n'.join(train_csv))
        
        with open(self.output_dir / "test_data_export.csv", 'w', encoding='utf-8') as f:
            f.write('\n'.join(test_csv))
        
        # Alpaca-Format
        train_alpaca = self.generate_classification_format(train_data, "alpaca")
        test_alpaca = self.generate_classification_format(test_data, "alpaca")
        
        with open(self.output_dir / "train_alpaca.jsonl", 'w', encoding='utf-8') as f:
            f.write('\n'.join(train_alpaca))
        
        with open(self.output_dir / "test_alpaca.jsonl", 'w', encoding='utf-8') as f:
            f.write('\n'.join(test_alpaca))
        
        # Few-Shot-Beispiele
        few_shot_examples = self.generate_few_shot_examples()
        with open(self.output_dir / "few_shot_examples.json", 'w', encoding='utf-8') as f:
            json.dump(few_shot_examples, f, ensure_ascii=False, indent=2)
        
        # Prompt-Template
        template = self.create_prompt_template()
        with open(self.output_dir / "prompt_template.txt", 'w', encoding='utf-8') as f:
            f.write(template)
        
        # Metadaten
        metadata = {
            "total_annotations": len(self.annotations),
            "train_size": len(train_data),
            "test_size": len(test_data),
            "frame_categories": self.frame_categories,
            "few_shot_examples": len(few_shot_examples),
            "formats": ["jsonl", "csv", "alpaca"]
        }
        
        with open(self.output_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Training-Daten gespeichert in {self.output_dir}")
        print(f"  - train.jsonl / test.jsonl (JSONL-Format)")
        print(f"  - training_data_export.csv / test_data_export.csv (CSV-Format)")
        print(f"  - train_alpaca.jsonl / test_alpaca.jsonl (Alpaca-Format)")
        print(f"  - few_shot_examples.json (Few-Shot-Beispiele)")
        print(f"  - prompt_template.txt (Prompt-Template)")
        print(f"  - metadata.json (Metadaten)")
    
    def generate_statistics(self, train_data: List[Dict], test_data: List[Dict]):
        """Generiert detaillierte Statistiken"""
        print("\n" + "="*80)
        print("TRAINING DATA STATISTIKEN")
        print("="*80)
        
        # Gesamtstatistiken
        total = len(train_data) + len(test_data)
        print(f"Gesamt Annotationen:     {total:,}")
        print(f"Train-Set:              {len(train_data):,} ({len(train_data)/total*100:.1f}%)")
        print(f"Test-Set:                {len(test_data):,} ({len(test_data)/total*100:.1f}%)")
        
        # Frame-Verteilung
        print(f"\nFrame-Verteilung (Train):")
        train_counts = Counter(ann['frame_label'] for ann in train_data)
        for frame, count in train_counts.most_common():
            percentage = (count / len(train_data) * 100)
            print(f"  {frame:15}: {count:4} ({percentage:5.1f}%)")
        
        print(f"\nFrame-Verteilung (Test):")
        test_counts = Counter(ann['frame_label'] for ann in test_data)
        for frame, count in test_counts.most_common():
            percentage = (count / len(test_data) * 100)
            print(f"  {frame:15}: {count:4} ({percentage:5.1f}%)")
        
        # Confidence-Statistiken
        train_confidences = [ann['confidence'] for ann in train_data]
        test_confidences = [ann['confidence'] for ann in test_data]
        
        print(f"\nConfidence-Statistiken:")
        print(f"  Train - Durchschnitt: {sum(train_confidences)/len(train_confidences):.1f}")
        print(f"  Test  - Durchschnitt:  {sum(test_confidences)/len(test_confidences):.1f}")
        
        # Text-Längen
        train_lengths = [len(ann['chunk_text'].split()) for ann in train_data]
        test_lengths = [len(ann['chunk_text'].split()) for ann in test_data]
        
        print(f"\nText-Längen (Wörter):")
        print(f"  Train - Durchschnitt: {sum(train_lengths)/len(train_lengths):.1f}")
        print(f"  Test  - Durchschnitt:  {sum(test_lengths)/len(test_lengths):.1f}")
    
    def run(self, test_size: float = 0.2, random_seed: int = 42):
        """Hauptfunktion"""
        self.load_annotations()
        
        if len(self.annotations) < 10:
            print("Warnung: Sehr wenige Annotationen. Mindestens 10 empfohlen.")
        
        # Erstelle Train/Test-Split
        train_data, test_data = self.create_train_test_split(test_size, random_seed)
        
        # Speichere Training-Daten
        self.save_training_data(train_data, test_data)
        
        # Zeige Statistiken
        self.generate_statistics(train_data, test_data)

def main():
    parser = argparse.ArgumentParser(description='Generiert Training-Daten für Frame-Classification')
    parser.add_argument('--annotations', required=True, help='Pfad zur annotations.json Datei')
    parser.add_argument('--output-dir', required=True, help='Ausgabeverzeichnis für Training-Daten')
    parser.add_argument('--test-size', type=float, default=0.2, help='Anteil der Test-Daten (0.0-1.0)')
    parser.add_argument('--random-seed', type=int, default=42, help='Random Seed für Reproduzierbarkeit')
    
    args = parser.parse_args()
    
    # Prüfe Input-Datei
    if not Path(args.annotations).exists():
        print(f"✗ Annotations-Datei {args.annotations} nicht gefunden!")
        return
    
    # Erstelle Generator
    generator = TrainingDataGenerator(args.annotations, args.output_dir)
    
    try:
        generator.run(args.test_size, args.random_seed)
        print(f"\n✓ Training-Daten erfolgreich generiert!")
        print(f"\nNächste Schritte:")
        print(f"1. Überprüfe die generierten Dateien in {args.output_dir}")
        print(f"2. Verwende die Daten für Fine-Tuning deines bevorzugten LLMs")
        print(f"3. Teste das Modell mit den Test-Daten")
        
    except Exception as e:
        print(f"Fehler: {e}")
        raise

if __name__ == "__main__":
    main()
